def generate_title_string():
    """
    Returns a string that will be printed as a header of the program.
    """
  
    ascii_galaxy = [
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠒⠒⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⢀⣤⣶⡾⠿⠿⠿⠿⣿⣿⣶⣦⣄⠙⠷⣤⡀⠀⠀⠀⠀",
        "⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣷⣄⠘⢿⡄⠀⠀⠀",
        "⠀⢀⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠐⠂⠠⢄⡀⠈⢿⣿⣧⠈⢿⡄⠀⠀",
        "⢀⠏⠀⠀⠀⢀⠄⣀⣴⣾⠿⠛⠛⠛⠷⣦⡙⢦⠀⢻⣿⡆⠘⡇⠀⠀",
        "⠀⠀⠀⠀⡐⢁⣴⡿⠋⢀⠠⣠⠤⠒⠲⡜⣧⢸⠄⢸⣿⡇⠀⡇⠀⠀",
        "⠀⠀⠀⡼⠀⣾⡿⠁⣠⢃⡞⢁⢔⣆⠔⣰⠏⡼⠀⣸⣿⠃⢸⠃⠀⠀",
        "⠀⠀⢰⡇⢸⣿⡇⠀⡇⢸⡇⣇⣀⣠⠔⠫⠊⠀⣰⣿⠏⡠⠃⠀⠀⢀",
        "⠀⠀⢸⡇⠸⣿⣷⠀⢳⡈⢿⣦⣀⣀⣀⣠⣴⣾⠟⠁⠀⠀⠀⠀⢀⡎",
        "⠀⠀⠘⣷⠀⢻⣿⣧⠀⠙⠢⠌⢉⣛⠛⠋⠉⠀⠀⠀⠀⠀⠀⣠⠎⠀",
        "⠀⠀⠀⠹⣧⡀⠻⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠃⠀⠀",
        "⠀⠀⠀⠀⠈⠻⣤⡈⠻⢿⣿⣷⣦⣤⣤⣤⣤⣤⣴⡾⠛⠉⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠈⠙⠶⢤⣈⣉⠛⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
    ]

    title_string = r"""    * ▗▄▄▖  ▗▄▖ ▗▖   .▗▄▖ ▗▖  ▗▖▗▖  ▗▖   .▗▄▄▄▄▖ ▗▄▖  ▗▄▖ .
      ▐▌  .▐▌ ▐▌▐▌  .▐▌ ▐▌ ▝▚▞▘  ▝▚▞▘   .    ▗▞▘▐▌ ▐▌▐▌ ▐▌
    . ▐▌▝▜▌▐▛▀▜▌▐▌   ▐▛▀▜▌  ▐▌   .▐▌       ▗▞▘  ▐▌ ▐▌▐▌ ▐▌   *
      ▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▗▞▘▝▚▖  ▐▌  *   ▐▙▄▄▄▖▝▚▄▞▘▝▚▄▞▘ ."""

    ascii_height = len(ascii_galaxy)
    title_lines = title_string.splitlines()
    title_height = len(title_lines)

    top_padding = (ascii_height - title_height) // 2
    title_lines = [""] * top_padding + title_lines
    title_lines += [""] * (ascii_height - len(title_lines)) 

    final_output = "\n".join(f"{ghost}  {title}" for ghost, title in zip(ascii_galaxy, title_lines))
    return final_output