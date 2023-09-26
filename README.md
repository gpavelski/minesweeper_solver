# minesweeper_solver
A Python bot for playing minesweeper on Windows.

Requirements:

  * ahk (AutoHotkey library: used for clicking on the game tiles)

  * pillow (used for screenshots of the game and image processing)

  * pymem (used for reading a few memory addresses and deciding the size of the game matrix)


It relies on the [Arbiter](https://github.com/jkrshnmenon/arbiter) version of the game.
It should be installed previously.
It is also required to have Autohotkey installed for using the ahk library.

This code basically reads the screen, processes the image, obtains a game matrix, make a decision (based on logic and probability), then outputs an action. This action is a click performed by AHK.
The process iterates until the conclusion of the episode.
The image processing step is based on pixel-value evaluation.
