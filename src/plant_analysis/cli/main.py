from __future__ import annotations

import typer

from plant_analysis.cli.attitude import app as attitude_command
from plant_analysis.cli.chirp import app as chirp_command
from plant_analysis.cli.diagnose import app as diagnose_command
from plant_analysis.cli.synthesize import app as synthesize_command

app = typer.Typer(help="PX4 chirp-based plant analysis tools")
app.command("chirp")(chirp_command)
app.command("synthesize")(synthesize_command)
app.command("diagnose")(diagnose_command)
app.command("attitude")(attitude_command)


if __name__ == "__main__":
    app()
