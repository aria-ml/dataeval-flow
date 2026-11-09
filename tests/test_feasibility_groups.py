"""The independent units `feasibility` hands DataEval: one group per set of items that may share a source."""

import numpy as np
import pytest

from dataeval_flow.evaluators.performance._feasibility_groups import group_items


def test_without_a_factor_or_duplicates_each_item_is_its_own_group() -> None:
    grouping = group_items([0, 1, 2, 3], None, [])
    assert grouping.units.tolist() == [0, 1, 2, 3]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (4, 1, 0)


def test_items_a_view_repeats_share_a_group() -> None:
    grouping = group_items([0, 1, 0, 2, 1], None, [])
    assert grouping.units.tolist() == [0, 1, 0, 2, 1]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (3, 2, 0)


def test_a_factor_joins_the_items_that_share_a_value() -> None:
    grouping = group_items(range(5), ["a", "b", "a", "c", "b"], [])
    assert grouping.units.tolist() == [0, 1, 0, 2, 1]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (3, 2, 0)


def test_an_item_without_a_value_joins_no_one() -> None:
    """Two items of unknown source are not shown to share one, and not shown apart: the factor says nothing of them."""
    grouping = group_items(range(4), ["a", None, "a", None], [])
    assert grouping.units.tolist() == [0, 1, 0, 2]


def test_duplicates_join_their_members_and_the_joins_are_counted() -> None:
    grouping = group_items(range(6), None, [[0, 1, 2], [4, 5]])
    assert grouping.units.tolist() == [0, 0, 0, 1, 2, 2]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (3, 3, 3)


def test_a_duplicate_pair_joins_two_groups_the_factor_made() -> None:
    """The merges are those the duplicates caused: three groups by factor, two once one pair joins two of them."""
    grouping = group_items(range(6), ["a", "a", "b", "b", "c", "c"], [[1, 2]])
    assert grouping.units.tolist() == [0, 0, 0, 0, 1, 1]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (2, 4, 1)


def test_a_duplicate_inside_a_group_the_factor_made_merges_nothing() -> None:
    grouping = group_items(range(4), ["a", "a", "b", "b"], [[0, 1]])
    assert grouping.units.tolist() == [0, 0, 1, 1]
    assert grouping.duplicate_merges == 0


def test_groups_that_share_a_member_become_one() -> None:
    grouping = group_items(range(5), None, [[0, 1], [1, 2], [3, 4]])
    assert grouping.units.tolist() == [0, 0, 0, 1, 1]


def test_the_factor_and_a_view_and_duplicates_all_join_the_same_group() -> None:
    grouping = group_items([0, 1, 0, 2, 3], ["a", "b", "c", "c", "d"], [[1, 4]])
    # item 2 repeats item 0 (a view); item 3 shares a factor value with item 2; item 4 duplicates item 1.
    assert grouping.units.tolist() == [0, 1, 0, 0, 1]
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (2, 3, 1)


def test_the_order_of_the_duplicates_changes_nothing() -> None:
    duplicates = [[7, 0], [2, 9, 4], [4, 5], [3, 8]]
    expected = group_items(range(10), None, duplicates).units.tolist()
    rng = np.random.default_rng(0)
    for _ in range(20):
        shuffled = [rng.permutation(members).tolist() for members in duplicates]
        assert group_items(range(10), None, [shuffled[i] for i in rng.permutation(len(shuffled))]).units.tolist() == (
            expected
        )


def test_groups_are_numbered_by_the_first_item_each_holds() -> None:
    assert group_items(range(5), None, [[4, 1], [3, 0]]).units.tolist() == [0, 1, 2, 0, 1]


def test_no_items_make_no_groups() -> None:
    grouping = group_items([], None, [])
    assert grouping.units.tolist() == []
    assert (grouping.groups, grouping.largest_group, grouping.duplicate_merges) == (0, 0, 0)


def test_the_units_are_integers() -> None:
    assert group_items([5, 5, 7], None, []).units.dtype == np.intp


def test_a_factor_that_does_not_cover_the_items_is_refused() -> None:
    with pytest.raises(ValueError, match="3 values for 4 items"):
        group_items(range(4), ["a", "a", "b"], [])


def test_a_duplicate_naming_an_item_that_is_not_there_is_refused() -> None:
    with pytest.raises(ValueError, match="item 9"):
        group_items(range(4), None, [[1, 9]])
