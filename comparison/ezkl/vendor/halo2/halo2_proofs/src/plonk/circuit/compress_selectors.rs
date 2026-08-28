use super::Expression;
use ff::Field;

/// This describes a selector and where it is activated.
#[derive(Debug, Clone)]
pub struct SelectorDescription {
    /// The selector that this description references, by index.
    pub selector: usize,

    /// The vector of booleans defining which rows are active for this selector.
    pub activations: Vec<bool>,

    /// The maximum degree of a gate involving this selector, including the
    /// virtual selector itself. This means this will be at least 1 for any
    /// expression containing a simple selector, even if that selector is not
    /// multiplied by anything.
    pub max_degree: usize,
}

/// This describes the assigned combination of a particular selector as well as
/// the expression it should be substituted with.
#[derive(Debug, Clone)]
pub struct SelectorAssignment<F> {
    /// The selector that this structure references, by index.
    pub selector: usize,

    /// The combination this selector was assigned to
    pub combination_index: usize,

    /// The expression we wish to substitute with
    pub expression: Expression<F>,
}

/// This function takes a vector that defines each selector as well as a closure
/// used to allocate new fixed columns, and returns the assignment of each
/// combination as well as details about each selector assignment.
///
/// This function takes
/// * `selectors`, a vector of `SelectorDescription`s that describe each
///   selector
/// * `max_degree`, the maximum allowed degree of any gate
/// * `allocate_fixed_columns`, a closure that constructs a new fixed column and
///   queries it at Rotation::cur(), returning the expression
///
/// and returns `Vec<Vec<F>>` containing the assignment of each new fixed column
/// (which each correspond to a combination) as well as a vector of
/// `SelectorAssignment` that the caller can use to perform the necessary
/// substitutions to the constraint system.
///
/// This function is completely deterministic.
pub fn process<F: Field, E>(
    mut selectors: Vec<SelectorDescription>,
    max_degree: usize,
    mut allocate_fixed_column: E,
    return_polys: bool,
) -> (Vec<Vec<F>>, Vec<SelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    if selectors.is_empty() {
        // There is nothing to optimize.
        return (vec![], vec![]);
    }

    // The length of all provided selectors must be the same.
    let n = selectors[0].activations.len();
    assert!(selectors.iter().all(|a| a.activations.len() == n));

    let mut combination_assignments = vec![];
    let mut selector_assignments = vec![];

    // All provided selectors of degree 0 are assumed to be either concrete
    // selectors or do not appear in a gate. Let's address these first.
    selectors.retain(|selector| {
        if selector.max_degree == 0 {
            // This is a complex selector, or a selector that does not appear in any
            // gate constraint.
            let expression = allocate_fixed_column();

            let combination_assignment = if return_polys {
                selector
                    .activations
                    .iter()
                    .map(|b| if *b { F::ONE } else { F::ZERO })
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };

            let combination_index = combination_assignments.len();
            combination_assignments.push(combination_assignment);
            selector_assignments.push(SelectorAssignment {
                selector: selector.selector,
                combination_index,
                expression,
            });

            false
        } else {
            true
        }
    });

    // All of the remaining `selectors` are simple. Let's try to combine them.
    // First, we compute the exclusion matrix that has (j, k) = true if selector
    // j and selector k conflict -- that is, they are both enabled on the same
    // row. This matrix is symmetric and the diagonal entries are false, so we
    // only need to store the lower triangular entries.
    let mut exclusion_matrix = (0..selectors.len())
        .map(|i| vec![false; i])
        .collect::<Vec<_>>();

    for (i, rows) in selectors
        .iter()
        .map(|selector| &selector.activations)
        .enumerate()
    {
        // Loop over the selectors previous to this one
        for (j, other_selector) in selectors.iter().enumerate().take(i) {
            // Look at what selectors are active at the same row
            if rows
                .iter()
                .zip(other_selector.activations.iter())
                .any(|(l, r)| l & r)
            {
                // Mark them as incompatible
                exclusion_matrix[i][j] = true;
            }
        }
    }

    // Simple selectors that we've added to combinations already.
    let mut added = vec![false; selectors.len()];

    for (i, selector) in selectors.iter().enumerate() {
        if added[i] {
            continue;
        }
        added[i] = true;
        assert!(selector.max_degree <= max_degree);
        // This is used to keep track of the largest degree gate involved in the
        // combination so far. We subtract by one to omit the virtual selector
        // which will be substituted by the caller with the expression we give
        // them.
        let mut d = selector.max_degree - 1;
        let mut combination = vec![selector];
        let mut combination_added = vec![i];

        // Try to find other selectors that can join this one.
        'try_selectors: for (j, selector) in selectors.iter().enumerate().skip(i + 1) {
            if d + combination.len() == max_degree {
                // Short circuit; nothing can be added to this
                // combination.
                break 'try_selectors;
            }

            // Skip selectors that have been added to previous combinations
            if added[j] {
                continue 'try_selectors;
            }

            // Is this selector excluded from co-existing in the same
            // combination with any of the other selectors so far?
            for &i in combination_added.iter() {
                if exclusion_matrix[j][i] {
                    continue 'try_selectors;
                }
            }

            // Can the new selector join the combination? Reminder: we use
            // selector.max_degree - 1 to omit the influence of the virtual
            // selector on the degree, as it will be substituted.
            let new_d = std::cmp::max(d, selector.max_degree - 1);
            if new_d + combination.len() + 1 > max_degree {
                // Guess not.
                continue 'try_selectors;
            }

            d = new_d;
            combination.push(selector);
            combination_added.push(j);
            added[j] = true;
        }

        // Now, compute the selector and combination assignments.
        let mut combination_assignment = if return_polys {
            vec![F::ZERO; n]
        } else {
            vec![]
        };
        let combination_len = combination.len();
        let combination_index = combination_assignments.len();
        let query = allocate_fixed_column();

        let mut assigned_root = F::ONE;
        selector_assignments.extend(combination.into_iter().map(|selector| {
            // Compute the expression for substitution. This produces an expression of the
            // form
            //     q * Prod[i = 1..=combination_len, i != assigned_root](i - q)
            //
            // which is non-zero only on rows where `combination_assignment` is set to
            // `assigned_root`. In particular, rows set to 0 correspond to all selectors
            // being disabled.
            let mut expression = query.clone();
            let mut root = F::ONE;
            for _ in 0..combination_len {
                if root != assigned_root {
                    expression = expression * (Expression::Constant(root) - query.clone());
                }
                root += F::ONE;
            }

            // Update the combination assignment
            // if return polys is false the iteration will not run as it is empty
            for (combination, selector) in combination_assignment
                .iter_mut()
                .zip(selector.activations.iter())
            {
                // This will not overwrite another selector's activations because
                // we have ensured that selectors are disjoint.
                if *selector {
                    *combination = assigned_root;
                }
            }

            assigned_root += F::ONE;

            SelectorAssignment {
                selector: selector.selector,
                combination_index,
                expression,
            }
        }));
        // if return polys is false this is empty
        combination_assignments.push(combination_assignment);
    }

    (combination_assignments, selector_assignments)
}
