use super::{Expression, SelectorBitmap};
use ff::Field;
use std::collections::HashMap;

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
    /// Whether this selector is the one activation source for its alias group.
    /// Other members map to the same expression and root, but do not need to
    /// rewrite the same fixed-polynomial rows again.
    pub is_activation_representative: bool,
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

#[derive(Debug)]
struct SelectorAliasGroup {
    selectors: Vec<usize>,
    max_degree: usize,
    complex: bool,
}

#[derive(Debug)]
struct SelectorCompressionPlan {
    groups: Vec<SelectorAliasGroup>,
    /// Each entry is one fixed column's selectors, represented by alias-group
    /// indices. Complex groups are singletons; simple groups use the standard
    /// non-overlapping greedy compression over unique activation patterns.
    combinations: Vec<Vec<usize>>,
}

/// Selector compression over packed activation columns. Exact activation
/// aliases are collapsed within each selector class before the unchanged
/// simple-selector greedy grouping. The original selectors remain in each
/// alias group so every circuit expression receives a replacement.
fn selector_combinations(
    degrees: &[usize],
    max_degree: usize,
    mut conflicts: impl FnMut(usize, usize) -> bool,
    mut activation_fingerprint: impl FnMut(usize) -> u64,
    mut same_activations: impl FnMut(usize, usize) -> bool,
) -> SelectorCompressionPlan {
    let mut groups = Vec::<SelectorAliasGroup>::new();
    let mut complex_group_ids = Vec::new();
    let mut simple_group_ids = Vec::new();
    let mut complex_groups_by_hash: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut simple_groups_by_hash: HashMap<u64, Vec<usize>> = HashMap::new();

    // Identical complex/unused selectors share one fixed polynomial. Simple
    // selectors use a separate alias class so their exact Boolean polynomial
    // is represented once during conflict and degree checks. Fingerprints
    // only select candidate buckets; exact row-set equality decides aliases.
    for (selector, degree) in degrees.iter().enumerate() {
        let is_complex = *degree == 0;
        let fingerprint = activation_fingerprint(selector);
        let groups_by_hash = if is_complex {
            &mut complex_groups_by_hash
        } else {
            assert!(*degree <= max_degree);
            &mut simple_groups_by_hash
        };
        let matching_group = groups_by_hash.get(&fingerprint).and_then(|candidates| {
            candidates.iter().copied().find(|group_index| {
                let representative = groups[*group_index].selectors[0];
                same_activations(selector, representative)
            })
        });
        if let Some(group_index) = matching_group {
            groups[group_index].selectors.push(selector);
            groups[group_index].max_degree = groups[group_index].max_degree.max(*degree);
        } else {
            let group_index = groups.len();
            groups.push(SelectorAliasGroup {
                selectors: vec![selector],
                max_degree: *degree,
                complex: is_complex,
            });
            groups_by_hash
                .entry(fingerprint)
                .or_default()
                .push(group_index);
            if is_complex {
                complex_group_ids.push(group_index);
            } else {
                simple_group_ids.push(group_index);
            }
        }
    }

    let mut combinations = complex_group_ids
        .into_iter()
        .map(|group_index| vec![group_index])
        .collect::<Vec<_>>();
    if simple_group_ids.is_empty() {
        return SelectorCompressionPlan {
            groups,
            combinations,
        };
    }

    let mut exclusion_matrix = (0..simple_group_ids.len())
        .map(|i| vec![false; i])
        .collect::<Vec<_>>();
    for (i, group_index) in simple_group_ids.iter().copied().enumerate() {
        let selector = groups[group_index].selectors[0];
        for (j, other_group_index) in simple_group_ids.iter().copied().enumerate().take(i) {
            let other_selector = groups[other_group_index].selectors[0];
            exclusion_matrix[i][j] = conflicts(selector, other_selector);
        }
    }

    let mut added = vec![false; simple_group_ids.len()];
    for i in 0..simple_group_ids.len() {
        if added[i] {
            continue;
        }
        added[i] = true;
        let first_group = simple_group_ids[i];
        let mut degree = groups[first_group].max_degree - 1;
        let mut combination = vec![first_group];
        let mut combination_indices = vec![i];

        'try_selectors: for j in (i + 1)..simple_group_ids.len() {
            let group_index = simple_group_ids[j];
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
            let new_degree = std::cmp::max(degree, groups[group_index].max_degree - 1);
            if new_degree + combination.len() + 1 > max_degree {
                continue 'try_selectors;
            }
            degree = new_degree;
            combination.push(group_index);
            combination_indices.push(j);
            added[j] = true;
        }
        combinations.push(combination);
    }

    SelectorCompressionPlan {
        groups,
        combinations,
    }
}

/// Count the exact columns that the packed compression routine will produce,
/// using the same greedy grouping and degree checks as key generation.
pub(crate) fn count_selector_combinations(
    selectors: &[super::SelectorActivationSet],
    degrees: &[usize],
    max_degree: usize,
) -> usize {
    assert_eq!(selectors.len(), degrees.len());
    selector_combinations(
        degrees,
        max_degree,
        |left, right| selectors[left].has_intersection(&selectors[right]),
        |selector| selectors[selector].activation_fingerprint(),
        |left, right| selectors[left].same_activations(&selectors[right]),
    )
    .combinations
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

    let plan = selector_combinations(
        degrees,
        max_degree,
        |left, right| selectors[left].has_intersection(&selectors[right]),
        |selector| selectors[selector].activation_fingerprint(),
        |left, right| selectors[left].same_activations(&selectors[right]),
    );
    let combination_count = plan.combinations.len();
    let mut selector_assignments = Vec::with_capacity(selectors.len());
    for (combination_index, combination) in plan.combinations.iter().enumerate() {
        let query = allocate_fixed_column();
        if plan.groups[combination[0]].complex {
            let group = &plan.groups[combination[0]];
            debug_assert_eq!(combination.len(), 1);
            for (member_index, selector) in group.selectors.iter().copied().enumerate() {
                selector_assignments.push(PackedSelectorAssignment {
                    selector,
                    combination_index,
                    root_value: 1,
                    expression: query.clone(),
                    is_activation_representative: member_index == 0,
                });
            }
            continue;
        }
        let combination_len = combination.len();
        let mut assigned_root = F::ONE;
        let mut root_value = 1u64;
        for group_index in combination {
            assert!(root_value <= max_degree as u64);
            let group = &plan.groups[*group_index];
            debug_assert!(!group.complex);
            let mut expression = query.clone();
            let mut root = F::ONE;
            for _ in 0..combination_len {
                if root != assigned_root {
                    expression = expression * (Expression::Constant(root) - query.clone());
                }
                root += F::ONE;
            }
            for (member_index, selector) in group.selectors.iter().copied().enumerate() {
                selector_assignments.push(PackedSelectorAssignment {
                    selector,
                    combination_index,
                    root_value,
                    expression: expression.clone(),
                    is_activation_representative: member_index == 0,
                });
            }
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
    allocate_fixed_column: E,
) -> (usize, Vec<PackedSelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    process_activation_sets_with_lookup(
        ActivationSelectorLookup::Direct(selectors),
        degrees,
        max_degree,
        allocate_fixed_column,
    )
}

/// Compress selector activations from a dictionary plus original-selector
/// references. This is the VK-read path: it never expands the cache back into
/// one activation object per original selector.
pub(crate) fn process_activation_patterns<F: Field, E>(
    patterns: &[super::SelectorActivationSet],
    selector_pattern_indices: &[u32],
    degrees: &[usize],
    max_degree: usize,
    allocate_fixed_column: E,
) -> (usize, Vec<PackedSelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    process_activation_sets_with_lookup(
        ActivationSelectorLookup::Mapped {
            patterns,
            selector_pattern_indices,
        },
        degrees,
        max_degree,
        allocate_fixed_column,
    )
}

enum ActivationSelectorLookup<'a> {
    Direct(&'a [super::SelectorActivationSet]),
    Mapped {
        patterns: &'a [super::SelectorActivationSet],
        selector_pattern_indices: &'a [u32],
    },
}

impl ActivationSelectorLookup<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Direct(selectors) => selectors.len(),
            Self::Mapped {
                selector_pattern_indices,
                ..
            } => selector_pattern_indices.len(),
        }
    }

    fn get(&self, selector: usize) -> &super::SelectorActivationSet {
        match self {
            Self::Direct(selectors) => &selectors[selector],
            Self::Mapped {
                patterns,
                selector_pattern_indices,
            } => &patterns[selector_pattern_indices[selector] as usize],
        }
    }
}

fn process_activation_sets_with_lookup<F: Field, E>(
    selectors: ActivationSelectorLookup<'_>,
    degrees: &[usize],
    max_degree: usize,
    mut allocate_fixed_column: E,
) -> (usize, Vec<PackedSelectorAssignment<F>>)
where
    E: FnMut() -> Expression<F>,
{
    assert_eq!(selectors.len(), degrees.len());
    if selectors.len() == 0 {
        return (0, Vec::new());
    }
    let n = selectors.get(0).len();
    assert!((0..selectors.len()).all(|index| selectors.get(index).len() == n));

    let plan = selector_combinations(
        degrees,
        max_degree,
        |left, right| selectors.get(left).has_intersection(selectors.get(right)),
        |selector| selectors.get(selector).activation_fingerprint(),
        |left, right| selectors.get(left).same_activations(selectors.get(right)),
    );
    let combination_count = plan.combinations.len();
    let mut selector_assignments = Vec::with_capacity(selectors.len());
    for (combination_index, combination) in plan.combinations.iter().enumerate() {
        let query = allocate_fixed_column();
        if plan.groups[combination[0]].complex {
            let group = &plan.groups[combination[0]];
            debug_assert_eq!(combination.len(), 1);
            for (member_index, selector) in group.selectors.iter().copied().enumerate() {
                selector_assignments.push(PackedSelectorAssignment {
                    selector,
                    combination_index,
                    root_value: 1,
                    expression: query.clone(),
                    is_activation_representative: member_index == 0,
                });
            }
            continue;
        }
        let combination_len = combination.len();
        let mut assigned_root = F::ONE;
        let mut root_value = 1u64;
        for group_index in combination {
            assert!(root_value <= max_degree as u64);
            let group = &plan.groups[*group_index];
            debug_assert!(!group.complex);
            let mut expression = query.clone();
            let mut root = F::ONE;
            for _ in 0..combination_len {
                if root != assigned_root {
                    expression = expression * (Expression::Constant(root) - query.clone());
                }
                root += F::ONE;
            }
            for (member_index, selector) in group.selectors.iter().copied().enumerate() {
                selector_assignments.push(PackedSelectorAssignment {
                    selector,
                    combination_index,
                    root_value,
                    expression: expression.clone(),
                    is_activation_representative: member_index == 0,
                });
            }
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
    fn native_complex_selector_aliasing_requires_equal_activation_sets() {
        let mut run_backed = super::super::SelectorActivationSet::zeroed(128);
        for row in 10..80 {
            run_backed.insert(row).unwrap();
        }
        let mut bitmap_backed = super::super::SelectorActivationSet::zeroed(128);
        for row in (10..80).step_by(2) {
            bitmap_backed.insert(row).unwrap();
        }
        for row in (11..80).step_by(2) {
            bitmap_backed.insert(row).unwrap();
        }
        assert!(matches!(
            &run_backed,
            super::super::SelectorActivationSet::Runs { .. }
        ));
        assert!(matches!(
            &bitmap_backed,
            super::super::SelectorActivationSet::Bitmap(_)
        ));
        assert!(run_backed.same_activations(&bitmap_backed));
        assert_eq!(
            run_backed.activation_fingerprint(),
            bitmap_backed.activation_fingerprint()
        );

        let mut different = super::super::SelectorActivationSet::zeroed(128);
        for row in (11..81).step_by(2) {
            different.insert(row).unwrap();
        }
        for row in (12..81).step_by(2) {
            different.insert(row).unwrap();
        }
        assert!(matches!(
            &different,
            super::super::SelectorActivationSet::Bitmap(_)
        ));
        assert!(!run_backed.same_activations(&different));
        let empty_a = super::super::SelectorActivationSet::zeroed(128);
        let empty_b = super::super::SelectorActivationSet::zeroed(128);
        let simple_run_backed = run_backed.clone();
        let simple_bitmap_backed = bitmap_backed.clone();
        let mut simple_disjoint = super::super::SelectorActivationSet::zeroed(128);
        for row in 80..96 {
            simple_disjoint.insert(row).unwrap();
        }
        let selectors = vec![
            run_backed,
            bitmap_backed,
            different,
            empty_a,
            empty_b,
            simple_run_backed,
            simple_bitmap_backed,
            simple_disjoint,
        ];
        // The identical simple selectors have different degrees. Their
        // alias group must use the higher degree before the greedy packing
        // considers the disjoint final selector.
        let degrees = vec![0, 0, 0, 0, 0, 1, 2, 1];
        let mut query = 0;
        let (column_count, assignments) = process_activation_sets::<Fp, _>(
            &selectors,
            &degrees,
            2,
            || {
                let column = query;
                query += 1;
                Expression::Fixed(FixedQuery {
                    index: Some(column),
                    column_index: column,
                    rotation: Rotation::cur(),
                })
            },
        );
        assert_eq!(column_count, 5);
        assert_eq!(assignments.len(), selectors.len());
        let assignment_for = |selector| {
            assignments
                .iter()
                .find(|assignment| assignment.selector == selector)
                .unwrap()
        };
        assert_eq!(
            assignment_for(0).combination_index,
            assignment_for(1).combination_index
        );
        assert_eq!(assignment_for(0).root_value, 1);
        assert_eq!(assignment_for(1).root_value, 1);
        assert!(assignment_for(0).is_activation_representative);
        assert!(!assignment_for(1).is_activation_representative);
        assert_ne!(
            assignment_for(0).combination_index,
            assignment_for(2).combination_index
        );
        assert_eq!(
            assignment_for(3).combination_index,
            assignment_for(4).combination_index
        );
        assert!(assignment_for(3).is_activation_representative);
        assert!(!assignment_for(4).is_activation_representative);
        assert_eq!(
            assignment_for(5).combination_index,
            assignment_for(6).combination_index
        );
        assert_eq!(assignment_for(5).root_value, 1);
        assert_eq!(assignment_for(6).root_value, 1);
        assert_ne!(
            assignment_for(5).combination_index,
            assignment_for(7).combination_index
        );
        assert!(!assignment_for(5).is_activation_representative);
        assert!(assignment_for(6).is_activation_representative);
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
