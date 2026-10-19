"""Fresh final fixtures for learned prediction plus integer residual coding.

These original topics have not appeared in the preceding experiments. Capture
only after the new source/model/configuration freeze. Numbered filler produces
correlated length variants; this is synthetic coverage rather than a corpus
benchmark or an independent task-quality evaluation.
"""


FINAL = [
    dict(
        name="residual_sourdough_bakery",
        kind="text",
        prefix="The first morning in a neighbourhood sourdough bakery.\n",
        body=(
            "Morning <cycle>: The baker lifted the linen cloth and checked "
            "the dough with a flour-covered finger. A tray of shaped loaves "
            "waited beside the oven while the apprentice swept the counter. "
            "They compared the crust of yesterday's bread with the longer "
            "fermentation planned for today. Soon the room filled with the "
            "smell of toasted flour, and the first customers gathered "
            "outside the open door. "
        ),
        suffix="\nBefore the next batch entered the oven, the baker decided to",
    ),
    dict(
        name="residual_ruby_leap_year",
        kind="code",
        prefix="Ruby code for checking Gregorian leap years.\n",
        body=(
            "# example <cycle>\n"
            "def leap_year?(year)\n"
            "  return true if year % 400 == 0\n"
            "  return false if year % 100 == 0\n"
            "  year % 4 == 0\n"
            "end\n"
            "raise 'expected leap year' unless leap_year?(2000)\n"
            "raise 'expected common year' if leap_year?(1900)\n"
            "# Test the century rule before the general four-year rule.\n"
        ),
        suffix="\n# Explain why the order of the two early returns matters.\n",
    ),
    dict(
        name="residual_swimming_championship",
        kind="structured",
        prefix="Regional swimming championship heat results.\n",
        body=(
            "Heat <cycle>: event=freestyle; distance_m=200; lane=4; "
            "swimmer=Amara; club=Harbour; time_ms=128460; rank=2; "
            "official=Ravi. The electronic timer is checked against "
            "the finish judge before results are confirmed.\n"
        ),
        suffix=(
            "\nFINAL RESULT: heat=BX-724; event=butterfly; distance_m=100; "
            "lane=6; swimmer=Leon; rank=1; verification=KESTREL-59.\n"
            "Question: What is the verification for heat BX-724?\n"
            "Answer:"
        ),
    ),
]


def fixture_manifest():
    return {
        "namespace": "kv_residual",
        "provenance": "Original synthetic fixtures, new to this experiment",
        "fixture_names": [row["name"] for row in FINAL],
        "kinds": [row["kind"] for row in FINAL],
        "lengths": [512, 2048],
        "caveat": (
            "Three fresh topics yield six correlated prompt/length coverage "
            "cases, not a representative workload or task-quality benchmark."
        ),
    }
