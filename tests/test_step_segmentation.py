from data.step_segmentation import rejoin_steps, segment_steps


def test_segment_steps_splits_numbered_solution_and_rejoins():
    solution = "Step 1: Add 2 and 3.\nStep 2: Return 5."

    steps = segment_steps(solution)

    assert steps == ["Add 2 and 3.", "Return 5."]
    assert rejoin_steps(steps) == "Add 2 and 3.\nReturn 5."


def test_segment_steps_falls_back_to_paragraphs():
    solution = "Find the total.\n\nCheck the arithmetic."

    assert segment_steps(solution) == ["Find the total.", "Check the arithmetic."]
