# TUI Spreadsheet

A fast, keyboard-driven spreadsheet application for your terminal. Built with Rust and [ratatui](https://github.com/ratatui/ratatui).

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Vim-like navigation** — Move through cells with arrow keys or `hjkl`
- **Formula engine** — Support for `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `IF`, `ABS`, `ROUND`, `FLOOR`, `CEIL`, `CONCAT`, `LEN`
- **Cell references** — Standard A1 notation (e.g., `A1`, `B2:C10`)
- **Undo/Redo** — Full history support with `Ctrl+Z` / `Ctrl+Y`
- **CSV import/export** — Work with existing spreadsheet files
- **Copy/Paste** — Clipboard operations between cells
- **Adjustable columns** — Resize columns with `+` and `-`
- **Clean TUI interface** — Built with ratatui for a smooth terminal experience

## Installation

### From Source

```bash
git clone https://github.com/Divhanthelion/sheet.git
cd sheet
cargo build --release
```

The binary will be available at `./target/release/tui-spreadsheet`.

### Prerequisites

- Rust 1.70+ 
- A terminal with Unicode support

## Usage

```bash
# Start with a blank spreadsheet
cargo run

# Open an existing CSV file
cargo run -- path/to/file.csv
```

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| `↑` `↓` `←` `→` / `hjkl` | Move cursor |
| `Tab` | Move right |
| `Shift+Tab` | Move left |
| `PageUp` / `PageDown` | Scroll up/down |
| `Home` | Go to column A |
| `End` | Go to last column in row |
| `Ctrl+Home` | Jump to cell A1 |

### Editing

| Key | Action |
|-----|--------|
| `Enter` / `F2` | Edit current cell |
| `Esc` | Cancel edit |
| `Backspace` / `Delete` | Clear cell |

### File Operations

| Key | Action |
|-----|--------|
| `Ctrl+S` | Save to CSV |
| `Ctrl+O` | Open CSV file |
| `Ctrl+Q` | Quit (prompts if unsaved) |

### Clipboard & History

| Key | Action |
|-----|--------|
| `Ctrl+C` | Copy cell |
| `Ctrl+V` | Paste cell |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |

### View

| Key | Action |
|-----|--------|
| `+` / `=` | Increase column width |
| `-` | Decrease column width |

## Formula Syntax

Formulas begin with `=` and support the following functions:

### Aggregate Functions
- `=SUM(A1:A10)` — Sum of range
- `=AVG(A1:A10)` — Average of range
- `=MIN(A1:A10)` — Minimum value
- `=MAX(A1:A10)` — Maximum value
- `=COUNT(A1:A10)` — Count of numeric cells

### Math Functions
- `=ABS(A1)` — Absolute value
- `=ROUND(A1, 2)` — Round to N decimal places
- `=FLOOR(A1)` — Round down
- `=CEIL(A1)` — Round up

### Text Functions
- `=CONCAT(A1, " ", B1)` — Concatenate strings
- `=LEN(A1)` — String length

### Logic Functions
- `=IF(A1>0, 1, 0)` — Conditional

### Arithmetic & References
- `=A1+B1` — Basic arithmetic
- `=A1*2` — Constants
- `=A1>B1` — Comparisons (returns 1 or 0)

## Architecture

The codebase is organized into logical modules:

```
src/
├── app/           # Application state and main loop
├── command/       # Action types and undo/redo history
│   ├── action/
│   └── history/
├── core/          # Core spreadsheet logic
│   ├── address/   # A1 notation parsing/formatting
│   ├── cell/      # Cell types and values
│   ├── csv/       # Import/export
│   ├── formula_engine/  # Formula parsing & evaluation
│   ├── sheet/     # Sheet data structure
│   └── workbook/  # Multi-sheet support
└── ui/            # Terminal interface
    ├── formula_bar/
    ├── grid_view/
    └── terminal/
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built with [ratatui](https://github.com/ratatui/ratatui) — A Rust library for building rich terminal user interfaces
- Inspired by classic spreadsheet applications and modern terminal tools
