use super::{Expression, SelectorBitmap};
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

/// A selector replacement together with its exact integer root in the
/// compressed fixed-column encoding. The root is represented as a small
/// integer until one fixed column at a time is converted to field elements.
#[derive(Debug, Clone)]
pub(crate) struct PackedSelectorAssignment<F> {
    pub selector: usize,
    pub combination_index: usize,
    pub root_value: u64,
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

/// Selector compression over packed activation columns. The ordering,
/// greedy grouping, degree checks, and selector replacement expressions are
/// the same as `process`; only the row-set representation changes.
fn selector_combinations(
    degrees: &[usize],
    max_degree: usize,
    mut conflicts: impl FnMut(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut combinations = Vec::new();
    let simple: Vec<usize> = degrees
        .iter()
        .enumerate()
        .filter_map(|(selector, degree)| (*degree != 0).then_some(selector))
        .collect();

    // Complex and unused selectors are kept as independent columns, in their
    // original order, just as in Halo2's standard compression routine.
    for (selector, degree) in degrees.iter().enumerate() {
        if *degree == 0 {
            combinations.push(vec![selector]);
        }
    }
    if simple.is_empty() {
        return combinations;
    }

    let mut exclusion_matrix = (0..simple.len())
        .map(|i| vec![false; i])
        .collect::<Vec<_>>();
    for (i, selector) in simple.iter().copied().enumerate() {
        for (j, other_selector) in simple.iter().copied().enumerate().take(i) {
            exclusion_matrix[i][j] = conflicts(selector, other_selector);
        }
    }

    let mut added = vec![false; simple.len()];
    for i in 0..simple.len() {
        if added[i] {
            continue;
        }
        added[i] = true;
        let first_selector = simple[i];
        assert!(degrees[first_selector] <= max_degree);
        let mut degree = degrees[first_selector] - 1;
        let mut combination = vec![first_selector];
        let mut combination_indices = vec![i];

        'try_selectors: for j in (i + 1)..simple.len() {
            let selector = simple[j];
            if degree + combination.len() == max_degree {
                break 'try_selectors;
            }
            if added[j] {
                continue 'try_selectors;
            }
            for &member in &combination_indices {
                if exclusion_matrix[j][member] {
                    continue 'try_selectors;
                }
            }
            assert!(degrees[selector] <= max_degree);
            let new_degree = std::cmp::max(degree, degrees[selector] - 1);
            if new_degree + combination.len() + 1 > max_degree {
                continue 'try_selectors;
            }
            degree = new_degree;
            combination.push(selector);
            combination_indices.push(j);
            added[j] = true;
        }
        combinations.push(combination);
    }

    combinations
}

/// Count the exact columns that the packed compression routine will produce,
/// using the same greedy grouping and degree checks as key generation.
pub(crate) fn count_selector_combinations(
    selectors: &[super::SelectorActivationSet],
    degrees: &[usize],
    max_degree: usize,
) -> usize {
    assert_eq!(selectors.len(), degrees.len());
    selector_combinations(degrees, max_degree, |left, right| {
        selectors[left].has_intersection(&selectors[right])
    })
    .len()
}

pub(crate) fn process_bitmaps<F: Field, E>(
    selectors: &[SelectorBitmap],
    degrees: &[usize],
    max_degree: usize,
    mut allocate_fixed_column: E,
) -> (usize, Vec<PackedSelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    assert_eq!(selectors.len(), degrees.len());
    if selectors.is_empty() {
        return (0, Vec::new());
    }
    let n = selectors[0].len();
    assert!(selectors.iter().all(|selector| selector.len() == n));

    let combinations = selector_combinations(degrees, max_degree, |left, right| {
        selectors[left].has_intersection(&selectors[right])
    });
    let combination_count = combinations.len();
    let mut selector_assignments = Vec::with_capacity(selectors.len());
    for (combination_index, combination) in combinations.into_iter().enumerate() {
        let query = allocate_fixed_column();
        let combination_len = combination.len();
        let mut assigned_root = F::ONE;
        let mut root_value = 1u64;
        for selector in combination {
            assert!(root_value <= max_degree as u64);
            let mut expression = query.clone();
            let mut root = F::ONE;
            for _ in 0..combination_len {
                if root != assigned_root {
                    expression = expression * (Expression::Constant(root) - query.clone());
                }
                root += F::ONE;
            }
            selector_assignments.push(PackedSelectorAssignment {
                selector,
                combination_index,
                root_value,
                expression,
            });
            assigned_root += F::ONE;
            root_value = root_value
                .checked_add(1)
                .expect("selector root value overflow");
        }
    }

    assert_eq!(selector_assignments.len(), selectors.len());
    (combination_count, selector_assignments)
}

/// Compress the same selectors directly from their hybrid row-set backing.
/// This uses the exact greedy grouping rule and root ordering as
/// `process_bitmaps`, without expanding run-backed selectors into dense arrays.
pub(crate) fn process_activation_sets<F: Field, E>(
    selectors: &[super::SelectorActivationSet],
    degrees: &[usize],
    max_degree: usize,
    mut allocate_fixed_column: E,
) -> (usize, Vec<PackedSelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    assert_eq!(selectors.len(), degrees.len());
    if selectors.is_empty() {
        return (0, Vec::new());
    }
    let n = selectors[0].len();
    assert!(selectors.iter().all(|selector| selector.len() == n));

    let combinations = selector_combinations(degrees, max_degree, |left, right| {
        selectors[left].has_intersection(&selectors[right])
    });
    let combination_count = combinations.len();
    let mut selector_assignments = Vec::with_capacity(selectors.len());
    for (combination_index, combination) in combinations.into_iter().enumerate() {
        let query = allocate_fixed_column();
        let combination_len = combination.len();
        let mut assigned_root = F::ONE;
        let mut root_value = 1u64;
        for selector in combination {
            assert!(root_value <= max_degree as u64);
            let mut expression = query.clone();
            let mut root = F::ONE;
            for _ in 0..combination_len {
                if root != assigned_root {
                    expression = expression * (Expression::Constant(root) - query.clone());
                }
                root += F::ONE;
            }
            selector_assignments.push(PackedSelectorAssignment {
                selector,
                combination_index,
                root_value,
                expression,
            });
            assigned_root += F::ONE;
            root_value = root_value
                .checked_add(1)
                .expect("selector root value overflow");
        }
    }

    assert_eq!(selector_assignments.len(), selectors.len());
    (combination_count, selector_assignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{plonk::FixedQuery, poly::Rotation};
    use halo2curves::pasta::Fp;
    use proptest::collection::{vec, SizeRange};
    use proptest::prelude::*;

    prop_compose! {
        fn arb_selector(assignment_size: usize, max_degree: usize)
                       (degree in 0..max_degree,
                        assignment in vec(any::<bool>(), assignment_size))
                       -> (usize, Vec<bool>) {
            (degree, assignment)
        }
    }

    prop_compose! {
        fn arb_selector_list(assignment_size: usize, max_degree: usize, num_selectors: impl Into<SizeRange>)
                            (list in vec(arb_selector(assignment_size, max_degree), num_selectors))
                            -> Vec<SelectorDescription>
        {
            list.into_iter().enumerate().map(|(i, (max_degree, activations))| {
                SelectorDescription {
                    selector: i,
                    activations,
                    max_degree,
                }
            }).collect()
        }
    }

    prop_compose! {
        fn arb_instance(max_assignment_size: usize,
                        max_degree: usize,
                        max_selectors: usize)
                       (assignment_size in 1..max_assignment_size,
                        degree in 1..max_degree,
                        num_selectors in 1..max_selectors)
                       (list in arb_selector_list(assignment_size, degree, num_selectors),
                        degree in Just(degree))
                       -> (Vec<SelectorDescription>, usize)
        {
            (list, degree)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10000))]
        #[test]
        fn test_selector_combination((selectors, max_degree) in arb_instance(10, 10, 15)) {
            let mut query = 0;
            let (combination_assignments, selector_assignments) =
                process::<Fp, _>(selectors.clone(), max_degree, || {
                    let tmp = Expression::Fixed(FixedQuery {
                        index: Some(query),
                        column_index: query,
                        rotation: Rotation::cur(),
                    });
                    query += 1;
                    tmp
                },  true);

            {
                let mut selectors_seen = vec![];
                assert_eq!(selectors.len(), selector_assignments.len());
                for selector in &selector_assignments {
                    // Every selector should be assigned to a combination
                    assert!(selector.combination_index < combination_assignments.len());
                    assert!(!selectors_seen.contains(&selector.selector));
                    selectors_seen.push(selector.selector);
                }
            }

            // Test that, for each selector, the provided expression
            //  1. evaluates to zero on rows where the selector's activation is off
            //  2. evaluates to nonzero on rows where the selector's activation is on
            //  3. is of degree d such that d + (selector.max_degree - 1) <= max_degree
            //     OR selector.max_degree is zero
            for selector in selector_assignments {
                assert_eq!(
                    selectors[selector.selector].activations.len(),
                    combination_assignments[selector.combination_index].len()
                );
                for (&activation, &assignment) in selectors[selector.selector]
                    .activations
                    .iter()
                    .zip(combination_assignments[selector.combination_index].iter())
                {
                    let eval = selector.expression.evaluate(
                        &|c| c,
                        &|_| panic!("should not occur in returned expressions"),
                        &|query| {
                            // Should be the correct combination in the expression
                            assert_eq!(selector.combination_index, query.index.unwrap());
                            assignment
                        },
                        &|_| panic!("should not occur in returned expressions"),
                        &|_| panic!("should not occur in returned expressions"),
                        &|_| panic!("should not occur in returned expressions"),
                        &|a| -a,
                        &|a, b| a + b,
                        &|a, b| a * b,
                        &|a, f| a * f,
                    );

                    if activation {
                        assert!(!eval.is_zero_vartime());
                    } else {
                        assert!(eval.is_zero_vartime());
                    }
                }

                let expr_degree = selector.expression.degree();
                assert!(expr_degree <= max_degree);
                if selectors[selector.selector].max_degree > 0 {
                    assert!(
                        (selectors[selector.selector].max_degree - 1) + expr_degree <= max_degree
                    );
                }
            }
        }
    }

    #[test]
    fn native_packed_selector_compression_preserves_group_values_and_order() {
        let mut raw = vec![vec![false; 256]; 5];
        raw[0][10..40].fill(true);
        raw[1][80..110].fill(true);
        for row in [1, 5, 9, 13, 17] {
            raw[2][row] = true;
        }
        raw[3][250] = true;
        raw[4][150] = true;
        let degrees = vec![1, 1, 1, 0, 1];
        let descriptions: Vec<_> = raw
            .iter()
            .zip(&degrees)
            .enumerate()
            .map(
                |(selector, (activations, max_degree))| SelectorDescription {
                    selector,
                    activations: activations.clone(),
                    max_degree: *max_degree,
                },
            )
            .collect();
        let mut old_query = 0;
        let (expected_columns, _) = process::<Fp, _>(
            descriptions,
            6,
            || {
                let query = old_query;
                old_query += 1;
                Expression::Fixed(FixedQuery {
                    index: Some(query),
                    column_index: query,
                    rotation: Rotation::cur(),
                })
            },
            true,
        );

        let packed = raw
            .iter()
            .map(|activations| {
                let mut bitmap = SelectorBitmap::zeroed(activations.len(), 1).unwrap();
                for (row, active) in activations.iter().copied().enumerate() {
                    if active {
                        bitmap.set_value(row, 1).unwrap();
                    }
                }
                bitmap
            })
            .collect::<Vec<_>>();
        let hybrid = raw
            .iter()
            .map(|activations| {
                let mut rows = super::super::SelectorActivationSet::zeroed(activations.len());
                for (row, active) in activations.iter().copied().enumerate() {
                    if active {
                        rows.insert(row).unwrap();
                    }
                }
                rows
            })
            .collect::<Vec<_>>();
        assert!(matches!(&hybrid[3], super::super::SelectorActivationSet::Runs { .. }));
        assert!(matches!(&hybrid[0], super::super::SelectorActivationSet::Bitmap(_)));
        for (expected, set) in packed.iter().zip(&hybrid) {
            let mut encoded = Vec::new();
            set.write_activation(&mut encoded).unwrap();
            assert_eq!(encoded, expected.as_bytes());
            let mut bytes = encoded.as_slice();
            let decoded = super::super::SelectorActivationSet::read_activation(
                &mut bytes,
                expected.len(),
            )
            .unwrap();
            let mut reencoded = Vec::new();
            decoded.write_activation(&mut reencoded).unwrap();
            assert_eq!(reencoded, expected.as_bytes());
        }
        let mut new_query = 0;
        let (new_column_count, assignments) =
            process_bitmaps::<Fp, _>(&packed, &degrees, 6, || {
                let query = new_query;
                new_query += 1;
                Expression::Fixed(FixedQuery {
                    index: Some(query),
                    column_index: query,
                    rotation: Rotation::cur(),
                })
            });
        assert_eq!(new_column_count, expected_columns.len());

        let assignments_by_column: Vec<Vec<_>> = (0..new_column_count)
            .map(|column| {
                assignments
                    .iter()
                    .filter(|assignment| assignment.combination_index == column)
                    .collect()
            })
            .collect();
        for (column_index, expected) in expected_columns.iter().enumerate() {
            let members = &assignments_by_column[column_index];
            let max_root = members
                .iter()
                .map(|assignment| assignment.root_value)
                .max()
                .unwrap();
            let width = u64::BITS - max_root.leading_zeros();
            let mut packed_column = SelectorBitmap::zeroed(raw[0].len(), width).unwrap();
            for assignment in members {
                packed[assignment.selector].for_each_set_bit(|row| {
                    packed_column.set_value(row, assignment.root_value).unwrap();
                });
            }
            let mut roots = vec![Fp::ZERO; max_root as usize + 1];
            for root in 1..roots.len() {
                roots[root] = roots[root - 1] + Fp::ONE;
            }
            for row in 0..packed_column.len() {
                assert_eq!(roots[packed_column.value(row) as usize], expected[row]);
            }
        }

        let mut hybrid_query = 0;
        let (hybrid_column_count, hybrid_assignments) =
            process_activation_sets::<Fp, _>(&hybrid, &degrees, 6, || {
                let query = hybrid_query;
                hybrid_query += 1;
                Expression::Fixed(FixedQuery {
                    index: Some(query),
                    column_index: query,
                    rotation: Rotation::cur(),
                })
            });
        assert_eq!(hybrid_column_count, new_column_count);
        for assignment in &hybrid_assignments {
            let packed_assignment = assignments
                .iter()
                .find(|other| other.selector == assignment.selector)
                .unwrap();
            assert_eq!(assignment.combination_index, packed_assignment.combination_index);
            assert_eq!(assignment.root_value, packed_assignment.root_value);
        }
        for (column_index, expected) in expected_columns.iter().enumerate() {
            let members = hybrid_assignments
                .iter()
                .filter(|assignment| assignment.combination_index == column_index)
                .collect::<Vec<_>>();
            let max_root = members
                .iter()
                .map(|assignment| assignment.root_value)
                .max()
                .unwrap();
            let width = u64::BITS - max_root.leading_zeros();
            let mut encoded = SelectorBitmap::zeroed(raw[0].len(), width).unwrap();
            for assignment in members {
                hybrid[assignment.selector].for_each_set_bit(|row| {
                    encoded.set_value(row, assignment.root_value).unwrap();
                });
            }
            let mut roots = vec![Fp::ZERO; max_root as usize + 1];
            for root in 1..roots.len() {
                roots[root] = roots[root - 1] + Fp::ONE;
            }
            for row in 0..encoded.len() {
                assert_eq!(roots[encoded.value(row) as usize], expected[row]);
            }
        }
    }

    #[test]
    fn native_hybrid_selector_stream_preserves_partial_byte_wire_encoding() {
        let mut packed = &[0x81u8, 0xf1][..];
        let selector = super::super::SelectorActivationSet::read_activation(&mut packed, 13).unwrap();
        let mut encoded = Vec::new();
        selector.write_activation(&mut encoded).unwrap();
        assert_eq!(encoded, [0x81, 0x11]);
    }
}
