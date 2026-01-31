//! A terminal spreadsheet application built with ratatui. Features include a navigable grid of cells,
//! formula support (SUM, AVG, MIN, MAX, COUNT, IF, ABS, ROUND, FLOOR, CEIL, CONCAT, LEN),
//! cell references in A1 notation, undo/redo, CSV import/export, copy/paste, and a formula bar.

pub mod app {

    use anyhow::{anyhow, Result};
    use crate::command::history::History;
    use crate::core::formula_engine::SimpleFormulaEngine;
    use crate::core::workbook::Workbook;
    use crate::ui::terminal::{AppState as TerminalAppState, TerminalApp};

    /// Shared mutable state for the application.
    pub struct AppState {
        /// The workbook containing all sheets.
        pub workbook: Workbook,
        /// History of actions for undo/redo support.
        pub history: History,
    }

    impl AppState {
        /// Create a new AppState with a default workbook.
        pub fn new() -> Self {
            let mut workbook = Workbook::new();
            workbook.add_sheet("Sheet1".to_string()).unwrap();
            Self {
                workbook,
                history: History::new(),
            }
        }
    }

    /// Entry point for the application.
    pub fn main_loop(app_state: &mut AppState, file_path: Option<String>) -> Result<()> {
        let mut ui = TerminalApp::new((80, 24))?;

        let sheet_name = "Sheet1";
        if app_state.workbook.get_sheet_mut(sheet_name).is_none() {
            app_state
                .workbook
                .add_sheet(sheet_name.to_string())
                .map_err(|e| anyhow!("Failed to add sheet: {:?}", e))?;
        }

        // If a CSV file path was provided, load it
        if let Some(ref path) = file_path {
            let p = std::path::Path::new(path);
            if p.exists() {
                let _ = crate::core::csv::import_csv(p, &mut app_state.workbook);
            }
        }

        let sheet = app_state
            .workbook
            .get_sheet_mut(sheet_name)
            .ok_or_else(|| anyhow!("Sheet '{}' not found", sheet_name))?;

        let mut terminal_state = TerminalAppState {
            sheet: sheet.clone(),
            engine: SimpleFormulaEngine,
            file_path,
        };

        ui.run(&mut terminal_state)?;

        *sheet = terminal_state.sheet;

        Ok(())
    }

}

pub mod command {
    pub mod action {

            use crate::core::address::Address;
            use crate::core::cell::CellValue;

            /// All possible commands.
            #[derive(Debug, Clone)]
            pub enum Action {
                SetCell(Address, CellValue),
                ClearCell(Address, CellValue),
            }

    }

    pub mod history {

            use crate::command::action::Action;

            /// Tracks executed actions for undo/redo.
            pub struct History {
                pub undo_stack: Vec<Action>,
                pub redo_stack: Vec<Action>,
            }

            impl History {
                /// Create a new empty History.
                pub fn new() -> Self {
                    Self {
                        undo_stack: Vec::new(),
                        redo_stack: Vec::new(),
                    }
                }

                /// Adds an action to the undo stack and clears redo.
                pub fn push(&mut self, action: Action) {
                    self.undo_stack.push(action);
                    self.redo_stack.clear();
                }

                /// Pops the last action for undo.
                pub fn undo(&mut self) -> Option<Action> {
                    self.undo_stack.pop()
                }

                /// Pops an action from the redo stack.
                pub fn redo(&mut self) -> Option<Action> {
                    self.redo_stack.pop()
                }

                /// Push onto redo stack.
                pub fn push_redo(&mut self, action: Action) {
                    self.redo_stack.push(action);
                }

                /// Push onto undo stack without clearing redo.
                pub fn push_undo(&mut self, action: Action) {
                    self.undo_stack.push(action);
                }
            }

    }

}

pub mod core {
    pub mod address {

            use std::fmt;

            /// Zero-based row and column indices.
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            pub struct Address {
                pub row: usize,
                pub col: usize,
            }

            /// Errors that can occur while parsing an address string.
            #[derive(Debug, Clone, PartialEq, Eq)]
            pub enum AddressParseError {
                InvalidFormat,
                InvalidColumn,
                InvalidRow,
            }

            impl fmt::Display for AddressParseError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        AddressParseError::InvalidFormat => write!(f, "invalid address format"),
                        AddressParseError::InvalidColumn => write!(f, "invalid column part"),
                        AddressParseError::InvalidRow => write!(f, "invalid row part"),
                    }
                }
            }

            impl std::error::Error for AddressParseError {}

            /// Parses strings like `"A1"` into an `Address`.
            pub fn parse_address(s: &str) -> Result<Address, AddressParseError> {
                if s.is_empty() {
                    return Err(AddressParseError::InvalidFormat);
                }

                let mut col_part = String::new();
                let mut row_part = String::new();

                for c in s.chars() {
                    if c.is_ascii_alphabetic() {
                        if !row_part.is_empty() {
                            return Err(AddressParseError::InvalidFormat);
                        }
                        col_part.push(c.to_ascii_uppercase());
                    } else if c.is_ascii_digit() {
                        row_part.push(c);
                    } else {
                        return Err(AddressParseError::InvalidFormat);
                    }
                }

                if col_part.is_empty() || row_part.is_empty() {
                    return Err(AddressParseError::InvalidFormat);
                }

                let mut col: usize = 0;
                for ch in col_part.chars() {
                    if !(ch >= 'A' && ch <= 'Z') {
                        return Err(AddressParseError::InvalidColumn);
                    }
                    col = col * 26 + ((ch as u8 - b'A' + 1) as usize);
                }
                col -= 1;

                let row: usize = match row_part.parse::<usize>() {
                    Ok(n) if n > 0 => n - 1,
                    _ => return Err(AddressParseError::InvalidRow),
                };

                Ok(Address { row, col })
            }

            /// Formats an `Address` back into A1 notation.
            pub fn format_address(addr: &Address) -> String {
                let mut col = addr.col + 1;
                let mut letters = Vec::new();
                while col > 0 {
                    let rem = (col - 1) % 26;
                    letters.push((b'A' + rem as u8) as char);
                    col = (col - 1) / 26;
                }
                letters.reverse();
                let col_str: String = letters.into_iter().collect();
                let row_str = (addr.row + 1).to_string();
                format!("{}{}", col_str, row_str)
            }

    }

    pub mod cell {

            use crate::core::address::Address;

            /// The three kinds of cell content.
            #[derive(Debug, Clone)]
            pub enum CellValue {
                Number(f64),
                Text(String),
                Formula(String),
            }

            /// A cell in the spreadsheet.
            #[derive(Debug, Clone)]
            pub struct Cell {
                pub address: Address,
                pub value: CellValue,
            }

    }

    pub mod formula_engine {

            use crate::core::sheet::Sheet;

            use std::fmt;
            use crate::core::cell::CellValue;
            use crate::core::address::{parse_address, Address};

            /// Errors that can occur while parsing or evaluating a formula.
            #[derive(Debug, Clone)]
            pub enum FormulaError {
                Parse(String),
                Eval(String),
            }

            impl fmt::Display for FormulaError {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    match self {
                        FormulaError::Parse(msg) => write!(f, "parse error: {}", msg),
                        FormulaError::Eval(msg) => write!(f, "evaluation error: {}", msg),
                    }
                }
            }

            impl std::error::Error for FormulaError {}

            /// Result type for formula evaluation: either a number or text.
            #[derive(Debug, Clone)]
            pub enum FormulaResult {
                Number(f64),
                Text(String),
            }

            /// Evaluates a formula string using the current sheet.
            pub trait FormulaEvaluator {
                fn evaluate(&self, formula: &str, sheet: &Sheet) -> Result<f64, FormulaError>;
                fn evaluate_rich(&self, formula: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError>;
            }

            /// A basic implementation of `FormulaEvaluator`.
            pub struct SimpleFormulaEngine;

            impl FormulaEvaluator for SimpleFormulaEngine {
                fn evaluate(&self, formula: &str, sheet: &Sheet) -> Result<f64, FormulaError> {
                    match self.evaluate_rich(formula, sheet)? {
                        FormulaResult::Number(n) => Ok(n),
                        FormulaResult::Text(_) => Err(FormulaError::Eval("expected number, got text".to_string())),
                    }
                }

                fn evaluate_rich(&self, formula: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError> {
                    let trimmed = formula.trim();

                    if trimmed.contains('(') {
                        return self.evaluate_function(trimmed, sheet);
                    }

                    // Try as arithmetic expression with cell refs and comparisons
                    Ok(FormulaResult::Number(self.evaluate_expr(trimmed, sheet)?))
                }
            }

            impl SimpleFormulaEngine {
                fn evaluate_function(&self, trimmed: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError> {
                    if !trimmed.ends_with(')') {
                        return Err(FormulaError::Parse("formula must end with ')'".to_string()));
                    }

                    let open_paren = trimmed
                        .find('(')
                        .ok_or_else(|| FormulaError::Parse("missing '('".to_string()))?;
                    let func_name = trimmed[..open_paren].to_uppercase();
                    let arg_str = trimmed[open_paren + 1..trimmed.len() - 1].trim();

                    match func_name.as_str() {
                        "IF" => return self.eval_if(arg_str, sheet),
                        "CONCAT" => return self.eval_concat(arg_str, sheet),
                        "LEN" => return self.eval_len(arg_str, sheet),
                        "ABS" => {
                            let val = self.evaluate_expr(arg_str, sheet)?;
                            return Ok(FormulaResult::Number(val.abs()));
                        }
                        "ROUND" => {
                            let parts = split_args(arg_str);
                            if parts.len() != 2 {
                                return Err(FormulaError::Parse("ROUND requires 2 arguments".to_string()));
                            }
                            let val = self.evaluate_expr(parts[0].trim(), sheet)?;
                            let digits = self.evaluate_expr(parts[1].trim(), sheet)? as i32;
                            let factor = 10f64.powi(digits);
                            return Ok(FormulaResult::Number((val * factor).round() / factor));
                        }
                        "FLOOR" => {
                            let val = self.evaluate_expr(arg_str, sheet)?;
                            return Ok(FormulaResult::Number(val.floor()));
                        }
                        "CEIL" => {
                            let val = self.evaluate_expr(arg_str, sheet)?;
                            return Ok(FormulaResult::Number(val.ceil()));
                        }
                        _ => {}
                    }

                    // Aggregate functions: SUM, AVG, MIN, MAX, COUNT
                    let cells = parse_range(arg_str)?;

                    let mut numbers: Vec<f64> = Vec::new();
                    for addr in cells {
                        if let Some(cell) = sheet.get_cell(&addr) {
                            match &cell.value {
                                CellValue::Number(v) => numbers.push(*v),
                                CellValue::Formula(f) => {
                                    if let Ok(v) = self.evaluate(f, sheet) {
                                        numbers.push(v);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }

                    let result = match func_name.as_str() {
                        "SUM" => Ok(numbers.iter().sum()),
                        "AVG" => {
                            if numbers.is_empty() {
                                Err(FormulaError::Eval("average of empty set".to_string()))
                            } else {
                                Ok(numbers.iter().sum::<f64>() / numbers.len() as f64)
                            }
                        }
                        "MIN" => {
                            if let Some(&min) = numbers.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                                Ok(min)
                            } else {
                                Err(FormulaError::Eval("min of empty set".to_string()))
                            }
                        }
                        "MAX" => {
                            if let Some(&max) = numbers.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                                Ok(max)
                            } else {
                                Err(FormulaError::Eval("max of empty set".to_string()))
                            }
                        }
                        "COUNT" => Ok(numbers.len() as f64),
                        _ => Err(FormulaError::Parse(format!("unknown function '{}'", func_name))),
                    }?;

                    Ok(FormulaResult::Number(result))
                }

                fn eval_if(&self, arg_str: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError> {
                    let parts = split_args(arg_str);
                    if parts.len() != 3 {
                        return Err(FormulaError::Parse("IF requires 3 arguments".to_string()));
                    }
                    let cond = self.evaluate_expr(parts[0].trim(), sheet)?;
                    if cond != 0.0 {
                        Ok(FormulaResult::Number(self.evaluate_expr(parts[1].trim(), sheet)?))
                    } else {
                        Ok(FormulaResult::Number(self.evaluate_expr(parts[2].trim(), sheet)?))
                    }
                }

                fn eval_concat(&self, arg_str: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError> {
                    let parts = split_args(arg_str);
                    let mut result = String::new();
                    for part in parts {
                        let trimmed = part.trim();
                        if let Ok(addr) = parse_address(trimmed) {
                            if let Some(cell) = sheet.get_cell(&addr) {
                                match &cell.value {
                                    CellValue::Text(t) => result.push_str(t),
                                    CellValue::Number(n) => result.push_str(&format_number(*n)),
                                    CellValue::Formula(f) => {
                                        match self.evaluate_rich(f, sheet) {
                                            Ok(FormulaResult::Number(n)) => result.push_str(&format_number(n)),
                                            Ok(FormulaResult::Text(t)) => result.push_str(&t),
                                            Err(_) => result.push_str("ERR"),
                                        }
                                    }
                                }
                            }
                        } else {
                            // Try as quoted string or number
                            let s = trimmed.trim_matches('"');
                            result.push_str(s);
                        }
                    }
                    Ok(FormulaResult::Text(result))
                }

                fn eval_len(&self, arg_str: &str, sheet: &Sheet) -> Result<FormulaResult, FormulaError> {
                    let trimmed = arg_str.trim();
                    if let Ok(addr) = parse_address(trimmed) {
                        if let Some(cell) = sheet.get_cell(&addr) {
                            let len = match &cell.value {
                                CellValue::Text(t) => t.len(),
                                CellValue::Number(n) => format_number(*n).len(),
                                CellValue::Formula(f) => {
                                    match self.evaluate_rich(f, sheet) {
                                        Ok(FormulaResult::Text(t)) => t.len(),
                                        Ok(FormulaResult::Number(n)) => format_number(n).len(),
                                        Err(_) => 0,
                                    }
                                }
                            };
                            Ok(FormulaResult::Number(len as f64))
                        } else {
                            Ok(FormulaResult::Number(0.0))
                        }
                    } else {
                        Ok(FormulaResult::Number(trimmed.trim_matches('"').len() as f64))
                    }
                }

                /// Evaluate an arithmetic expression with +, -, *, /, comparisons and cell references.
                fn evaluate_expr(&self, expr: &str, sheet: &Sheet) -> Result<f64, FormulaError> {
                    let tokens = tokenize(expr)?;
                    eval_comparison(&tokens, &mut 0, sheet)
                }
            }

            fn format_number(n: f64) -> String {
                if n == (n as i64) as f64 {
                    format!("{}", n as i64)
                } else {
                    format!("{:.2}", n)
                }
            }

            /// Split arguments by comma, respecting parentheses depth.
            fn split_args(s: &str) -> Vec<&str> {
                let mut parts = Vec::new();
                let mut depth = 0usize;
                let mut start = 0;
                for (i, c) in s.char_indices() {
                    match c {
                        '(' => depth += 1,
                        ')' => { if depth > 0 { depth -= 1; } }
                        ',' if depth == 0 => {
                            parts.push(&s[start..i]);
                            start = i + 1;
                        }
                        _ => {}
                    }
                }
                parts.push(&s[start..]);
                parts
            }

            #[derive(Debug, Clone)]
            enum Token {
                Number(f64),
                CellRef(Address),
                Op(char),
                Cmp(String), // >=, <=, ==, !=, >, <
                LParen,
                RParen,
            }

            fn tokenize(expr: &str) -> Result<Vec<Token>, FormulaError> {
                let mut tokens = Vec::new();
                let chars: Vec<char> = expr.chars().collect();
                let mut i = 0;
                while i < chars.len() {
                    let c = chars[i];
                    if c.is_whitespace() {
                        i += 1;
                        continue;
                    }
                    if c == '(' {
                        tokens.push(Token::LParen);
                        i += 1;
                    } else if c == ')' {
                        tokens.push(Token::RParen);
                        i += 1;
                    } else if (c == '>' || c == '<' || c == '!' || c == '=') && {
                        // Check if this is a comparison operator
                        c == '>' || c == '<' || (c == '!' && i + 1 < chars.len() && chars[i + 1] == '=')
                            || (c == '=' && i + 1 < chars.len() && chars[i + 1] == '=')
                    } {
                        let mut op = String::new();
                        op.push(c);
                        i += 1;
                        if i < chars.len() && chars[i] == '=' {
                            op.push('=');
                            i += 1;
                        }
                        tokens.push(Token::Cmp(op));
                    } else if c == '+' || c == '-' || c == '*' || c == '/' {
                        if c == '-' && (tokens.is_empty() || matches!(tokens.last(), Some(Token::Op(_)) | Some(Token::LParen) | Some(Token::Cmp(_)))) {
                            i += 1;
                            let start = i;
                            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                                i += 1;
                            }
                            if i == start {
                                return Err(FormulaError::Parse("expected number after '-'".to_string()));
                            }
                            let s: String = chars[start..i].iter().collect();
                            let n: f64 = s.parse().map_err(|_| FormulaError::Parse(format!("invalid number '-{}'", s)))?;
                            tokens.push(Token::Number(-n));
                        } else {
                            tokens.push(Token::Op(c));
                            i += 1;
                        }
                    } else if c.is_ascii_digit() || c == '.' {
                        let start = i;
                        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                            i += 1;
                        }
                        let s: String = chars[start..i].iter().collect();
                        let n: f64 = s.parse().map_err(|_| FormulaError::Parse(format!("invalid number '{}'", s)))?;
                        tokens.push(Token::Number(n));
                    } else if c.is_ascii_alphabetic() {
                        let start = i;
                        while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                            i += 1;
                        }
                        let s: String = chars[start..i].iter().collect();
                        match parse_address(&s) {
                            Ok(addr) => tokens.push(Token::CellRef(addr)),
                            Err(_) => return Err(FormulaError::Parse(format!("unknown identifier '{}'", s))),
                        }
                    } else {
                        return Err(FormulaError::Parse(format!("unexpected character '{}'", c)));
                    }
                }
                Ok(tokens)
            }

            fn resolve_value(token: &Token, sheet: &Sheet) -> Result<f64, FormulaError> {
                match token {
                    Token::Number(n) => Ok(*n),
                    Token::CellRef(addr) => {
                        match sheet.get_cell(addr) {
                            Some(cell) => match &cell.value {
                                CellValue::Number(n) => Ok(*n),
                                CellValue::Formula(f) => {
                                    let engine = SimpleFormulaEngine;
                                    engine.evaluate(f, sheet)
                                }
                                _ => Ok(0.0),
                            },
                            None => Ok(0.0),
                        }
                    }
                    _ => Err(FormulaError::Parse("expected a value".to_string())),
                }
            }

            /// Parse comparison expressions: additive (('>'|'<'|'>='|'<='|'=='|'!=') additive)?
            fn eval_comparison(tokens: &[Token], pos: &mut usize, sheet: &Sheet) -> Result<f64, FormulaError> {
                let left = eval_additive(tokens, pos, sheet)?;
                if *pos < tokens.len() {
                    if let Token::Cmp(op) = &tokens[*pos] {
                        let op = op.clone();
                        *pos += 1;
                        let right = eval_additive(tokens, pos, sheet)?;
                        let result = match op.as_str() {
                            ">" => left > right,
                            "<" => left < right,
                            ">=" => left >= right,
                            "<=" => left <= right,
                            "==" => (left - right).abs() < f64::EPSILON,
                            "!=" => (left - right).abs() >= f64::EPSILON,
                            _ => return Err(FormulaError::Parse(format!("unknown operator '{}'", op))),
                        };
                        return Ok(if result { 1.0 } else { 0.0 });
                    }
                }
                Ok(left)
            }

            /// Parse additive expressions: term (('+' | '-') term)*
            fn eval_additive(tokens: &[Token], pos: &mut usize, sheet: &Sheet) -> Result<f64, FormulaError> {
                let mut result = eval_multiplicative(tokens, pos, sheet)?;
                while *pos < tokens.len() {
                    match &tokens[*pos] {
                        Token::Op('+') => { *pos += 1; result += eval_multiplicative(tokens, pos, sheet)?; }
                        Token::Op('-') => { *pos += 1; result -= eval_multiplicative(tokens, pos, sheet)?; }
                        _ => break,
                    }
                }
                Ok(result)
            }

            /// Parse multiplicative expressions: atom (('*' | '/') atom)*
            fn eval_multiplicative(tokens: &[Token], pos: &mut usize, sheet: &Sheet) -> Result<f64, FormulaError> {
                let mut result = eval_atom(tokens, pos, sheet)?;
                while *pos < tokens.len() {
                    match &tokens[*pos] {
                        Token::Op('*') => { *pos += 1; result *= eval_atom(tokens, pos, sheet)?; }
                        Token::Op('/') => {
                            *pos += 1;
                            let divisor = eval_atom(tokens, pos, sheet)?;
                            if divisor == 0.0 {
                                return Err(FormulaError::Eval("division by zero".to_string()));
                            }
                            result /= divisor;
                        }
                        _ => break,
                    }
                }
                Ok(result)
            }

            /// Parse atoms: number, cell ref, or parenthesized expression
            fn eval_atom(tokens: &[Token], pos: &mut usize, sheet: &Sheet) -> Result<f64, FormulaError> {
                if *pos >= tokens.len() {
                    return Err(FormulaError::Parse("unexpected end of expression".to_string()));
                }
                match &tokens[*pos] {
                    Token::Number(_) | Token::CellRef(_) => {
                        let val = resolve_value(&tokens[*pos], sheet)?;
                        *pos += 1;
                        Ok(val)
                    }
                    Token::LParen => {
                        *pos += 1;
                        let val = eval_comparison(tokens, pos, sheet)?;
                        if *pos < tokens.len() && matches!(&tokens[*pos], Token::RParen) {
                            *pos += 1;
                        } else {
                            return Err(FormulaError::Parse("missing closing ')'".to_string()));
                        }
                        Ok(val)
                    }
                    _ => Err(FormulaError::Parse(format!("unexpected token {:?}", &tokens[*pos]))),
                }
            }

            /// Parse a range string (e.g. "A1:B2" or "C3") into a vector of `Address` values.
            fn parse_range(s: &str) -> Result<Vec<Address>, FormulaError> {
                let parts: Vec<&str> = s.split(':').collect();
                match parts.len() {
                    1 => {
                        let addr = parse_address(parts[0])
                            .map_err(|e| FormulaError::Parse(format!("invalid address '{}': {:?}", parts[0], e)))?;
                        Ok(vec![addr])
                    }
                    2 => {
                        let start = parse_address(parts[0])
                            .map_err(|e| FormulaError::Parse(format!("invalid start address '{}': {:?}", parts[0], e)))?;
                        let end = parse_address(parts[1])
                            .map_err(|e| FormulaError::Parse(format!("invalid end address '{}': {:?}", parts[1], e)))?;
                        if start.row > end.row || start.col > end.col {
                            return Err(FormulaError::Parse(
                                "range start must be top-left of end".to_string(),
                            ));
                        }
                        let mut cells = Vec::new();
                        for r in start.row..=end.row {
                            for c in start.col..=end.col {
                                cells.push(Address { row: r, col: c });
                            }
                        }
                        Ok(cells)
                    }
                    _ => Err(FormulaError::Parse(
                        "range must contain at most one ':'".to_string(),
                    )),
                }
            }

    }

    pub mod sheet {

            use std::collections::HashMap;

            use crate::core::address::Address;
            use crate::core::cell::{Cell, CellValue};
            use crate::core::formula_engine::FormulaError;

            /// A named collection of cells.
            #[derive(Clone)]
            pub struct Sheet {
                pub name: String,
                pub(crate) cells: HashMap<Address, Cell>,
            }

            /// Errors that can occur while working with a sheet.
            #[derive(Debug, Clone)]
            pub enum SheetError {
                Formula(FormulaError),
            }

            impl From<FormulaError> for SheetError {
                fn from(err: FormulaError) -> Self {
                    SheetError::Formula(err)
                }
            }

            impl std::fmt::Display for SheetError {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self {
                        SheetError::Formula(e) => write!(f, "formula error: {}", e),
                    }
                }
            }

            impl std::error::Error for SheetError {}

            impl Sheet {
                /// Retrieves a cell by address.
                pub fn get_cell(&self, addr: &Address) -> Option<&Cell> {
                    self.cells.get(addr)
                }

                /// Sets the value of a cell, inserting or updating it.
                pub fn set_cell_value(
                    &mut self,
                    addr: Address,
                    value: CellValue,
                ) -> Result<(), SheetError> {
                    let cell = Cell { address: addr, value };
                    self.cells.insert(addr, cell);
                    Ok(())
                }

                /// Removes a cell entirely.
                pub fn clear_cell(&mut self, addr: &Address) {
                    self.cells.remove(addr);
                }

                /// Returns the maximum occupied row index, or None if empty.
                pub fn max_occupied_col_in_row(&self, row: usize) -> Option<usize> {
                    self.cells.keys()
                        .filter(|a| a.row == row)
                        .map(|a| a.col)
                        .max()
                }
            }

    }

    pub mod workbook {

            use std::collections::HashMap;

            /// A collection of named sheets.
            pub struct Workbook {
                sheets: Vec<crate::core::sheet::Sheet>,
            }

            /// Errors that can occur while manipulating a workbook.
            #[derive(Debug, Clone)]
            pub enum WorkbookError {
                DuplicateSheetName(String),
            }

            impl Workbook {
                pub fn new() -> Self {
                    Workbook { sheets: Vec::new() }
                }

                pub fn add_sheet(&mut self, name: String) -> Result<(), WorkbookError> {
                    if self.sheets.iter().any(|s| s.name == name) {
                        return Err(WorkbookError::DuplicateSheetName(name));
                    }
                    let sheet = crate::core::sheet::Sheet {
                        name,
                        cells: HashMap::new(),
                    };
                    self.sheets.push(sheet);
                    Ok(())
                }

                pub fn get_sheet_mut(&mut self, name: &str) -> Option<&mut crate::core::sheet::Sheet> {
                    self.sheets.iter_mut().find(|s| s.name == name)
                }
            }

    }

    pub mod csv {

            use std::fs::File;
            use std::io::{self, BufRead, BufWriter, Write};
            use std::path::Path;

            use crate::core::address::Address;
            use crate::core::cell::{Cell, CellValue};
            use crate::core::sheet::Sheet;
            use crate::core::workbook::Workbook;

            /// Errors that can occur when importing/exporting CSV.
            #[derive(Debug)]
            pub enum CsvError {
                Io(io::Error),
                Parse(String),
                NoFirstSheet,
            }

            impl From<io::Error> for CsvError {
                fn from(err: io::Error) -> Self { CsvError::Io(err) }
            }

            /// Loads a CSV into the first sheet of the workbook.
            pub fn import_csv(path: &Path, workbook: &mut Workbook) -> Result<(), CsvError> {
                let sheet = workbook
                    .get_sheet_mut("Sheet1")
                    .ok_or(CsvError::NoFirstSheet)?;

                let sheet_cells = &mut sheet.cells;
                sheet_cells.clear();

                let file = File::open(path)?;
                let reader = io::BufReader::new(file);

                for (row_idx, line_res) in reader.lines().enumerate() {
                    let line: String = line_res?;
                    let parts: Vec<&str> = line.split(',').collect();
                    for (col_idx, part) in parts.iter().enumerate() {
                        let addr = Address { row: row_idx, col: col_idx };
                        let value = if part.starts_with('=') {
                            CellValue::Formula(part[1..].to_string())
                        } else if let Ok(num) = part.parse::<f64>() {
                            CellValue::Number(num)
                        } else {
                            CellValue::Text(part.to_string())
                        };
                        sheet_cells.insert(addr, Cell { address: addr, value });
                    }
                }

                Ok(())
            }

            /// Writes a sheet to CSV.
            pub fn export_csv(path: &Path, sheet: &Sheet) -> Result<(), CsvError> {
                let sheet_cells = &sheet.cells;

                let mut max_row = 0usize;
                let mut max_col = 0usize;
                for addr in sheet_cells.keys() {
                    if addr.row > max_row { max_row = addr.row; }
                    if addr.col > max_col { max_col = addr.col; }
                }

                if sheet_cells.is_empty() {
                    // Nothing to write
                    let _file = File::create(path)?;
                    return Ok(());
                }

                let mut rows: Vec<Vec<String>> = vec![vec!["".to_string(); max_col + 1]; max_row + 1];
                for (addr, cell) in sheet_cells {
                    let s = match &cell.value {
                        CellValue::Number(n) => n.to_string(),
                        CellValue::Text(t) => t.clone(),
                        CellValue::Formula(f) => format!("={}", f),
                    };
                    rows[addr.row][addr.col] = s;
                }

                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);

                for row in rows {
                    writeln!(writer, "{}", row.join(","))?;
                }

                Ok(())
            }

            /// Import CSV directly into a Sheet (not via workbook).
            pub fn import_csv_to_sheet(path: &Path, sheet: &mut Sheet) -> Result<(), CsvError> {
                sheet.cells.clear();

                let file = File::open(path)?;
                let reader = io::BufReader::new(file);

                for (row_idx, line_res) in reader.lines().enumerate() {
                    let line: String = line_res?;
                    let parts: Vec<&str> = line.split(',').collect();
                    for (col_idx, part) in parts.iter().enumerate() {
                        let addr = Address { row: row_idx, col: col_idx };
                        let value = if part.starts_with('=') {
                            CellValue::Formula(part[1..].to_string())
                        } else if let Ok(num) = part.parse::<f64>() {
                            CellValue::Number(num)
                        } else {
                            CellValue::Text(part.to_string())
                        };
                        sheet.cells.insert(addr, Cell { address: addr, value });
                    }
                }

                Ok(())
            }

    }

}

pub mod ui {
    pub mod grid_view {

            use std::cmp::max;

            use ratatui::{
                layout::Rect,
                style::{Color, Modifier, Style},
                text::{Line, Span},
                widgets::{Block, Borders, Paragraph},
            };

            use crate::core::address::{Address, format_address};
            use crate::core::cell::CellValue;
            use crate::core::formula_engine::{FormulaEvaluator, FormulaResult, SimpleFormulaEngine};
            use crate::core::sheet::Sheet;

            pub const DEFAULT_COL_WIDTH: usize = 10;
            pub const ROW_NUM_WIDTH: usize = 4;

            fn col_letter(col: usize) -> String {
                let mut c = col + 1;
                let mut letters = Vec::new();
                while c > 0 {
                    let rem = (c - 1) % 26;
                    letters.push((b'A' + rem as u8) as char);
                    c = (c - 1) / 26;
                }
                letters.reverse();
                letters.into_iter().collect()
            }

            /// Format a number for display, with large-number abbreviation.
            fn format_display_number(n: f64) -> String {
                let abs = n.abs();
                if abs >= 1_000_000_000.0 {
                    format!("{:.2}B", n / 1_000_000_000.0)
                } else if abs >= 1_000_000.0 {
                    format!("{:.2}M", n / 1_000_000.0)
                } else if n == (n as i64) as f64 {
                    format!("{}", n as i64)
                } else {
                    format!("{:.2}", n)
                }
            }

            pub struct GridView {
                pub cursor: Address,
                pub scroll_row: usize,
                pub scroll_col: usize,
                pub col_width: usize,
            }

            impl GridView {
                pub fn new(_viewport: (usize, usize)) -> Self {
                    GridView {
                        cursor: Address { row: 0, col: 0 },
                        scroll_row: 0,
                        scroll_col: 0,
                        col_width: DEFAULT_COL_WIDTH,
                    }
                }

                pub fn visible_dimensions(&self, area: Rect) -> (usize, usize) {
                    let inner_w = area.width.saturating_sub(2) as usize; // borders
                    let inner_h = area.height.saturating_sub(2) as usize; // borders
                    let visible_cols = inner_w.saturating_sub(ROW_NUM_WIDTH) / (self.col_width + 1); // +1 for separator
                    let visible_rows = inner_h.saturating_sub(2); // header + separator line
                    (visible_cols.max(1), visible_rows.max(1))
                }

                pub fn draw(&self, f: &mut ratatui::Frame, sheet: &Sheet, area: Rect) {
                    let block = Block::default().borders(Borders::ALL).title(format!(
                        " Sheet  [{}] ",
                        format_address(&self.cursor)
                    ));
                    let inner = block.inner(area);
                    f.render_widget(block, area);

                    if inner.width < 6 || inner.height < 3 {
                        return;
                    }

                    let col_w = self.col_width;
                    let visible_cols = (inner.width as usize).saturating_sub(ROW_NUM_WIDTH) / (col_w + 1);
                    let visible_rows = (inner.height as usize).saturating_sub(2); // header + separator

                    if visible_cols == 0 || visible_rows == 0 {
                        return;
                    }

                    let engine = SimpleFormulaEngine;
                    let mut lines: Vec<Line> = Vec::new();

                    // Header row: row-num gutter + column letters with separators
                    let mut header_spans = vec![Span::styled(
                        format!("{:>width$}", "", width = ROW_NUM_WIDTH),
                        Style::default().fg(Color::DarkGray),
                    )];
                    for c in 0..visible_cols {
                        let col_idx = self.scroll_col + c;
                        let label = col_letter(col_idx);
                        if c > 0 {
                            header_spans.push(Span::styled(
                                "│".to_string(),
                                Style::default().fg(Color::DarkGray),
                            ));
                        } else {
                            header_spans.push(Span::styled(" ", Style::default()));
                        }
                        header_spans.push(Span::styled(
                            format!("{:^width$}", label, width = col_w),
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                        ));
                    }
                    lines.push(Line::from(header_spans));

                    // Separator line under headers
                    let total_data_width = if visible_cols > 0 {
                        ROW_NUM_WIDTH + 1 + visible_cols * col_w + (visible_cols - 1)
                    } else {
                        ROW_NUM_WIDTH
                    };
                    let sep: String = format!(
                        "{:>width$}{}",
                        "",
                        "─".repeat(total_data_width.saturating_sub(ROW_NUM_WIDTH)),
                        width = ROW_NUM_WIDTH
                    );
                    lines.push(Line::from(Span::styled(
                        sep,
                        Style::default().fg(Color::DarkGray),
                    )));

                    // Data rows
                    for r in 0..visible_rows {
                        let row_idx = self.scroll_row + r;
                        let is_even_row = row_idx % 2 == 0;
                        let row_bg = if is_even_row {
                            Color::Rgb(30, 30, 30)
                        } else {
                            Color::Reset
                        };

                        let mut spans = vec![Span::styled(
                            format!("{:>width$}", row_idx + 1, width = ROW_NUM_WIDTH),
                            Style::default().fg(Color::DarkGray).bg(row_bg),
                        )];

                        for c in 0..visible_cols {
                            let col_idx = self.scroll_col + c;
                            let addr = Address { row: row_idx, col: col_idx };
                            let is_cursor = addr.row == self.cursor.row && addr.col == self.cursor.col;

                            // Separator between columns
                            if c > 0 {
                                spans.push(Span::styled(
                                    "│".to_string(),
                                    Style::default().fg(Color::Rgb(60, 60, 60)).bg(row_bg),
                                ));
                            } else {
                                spans.push(Span::styled(" ", Style::default().bg(row_bg)));
                            }

                            let (text, is_negative, is_err) = match sheet.get_cell(&addr) {
                                Some(cell) => match &cell.value {
                                    CellValue::Number(n) => {
                                        (format_display_number(*n), *n < 0.0, false)
                                    }
                                    CellValue::Text(t) => (t.clone(), false, false),
                                    CellValue::Formula(fml) => match engine.evaluate_rich(fml, sheet) {
                                        Ok(FormulaResult::Number(v)) => {
                                            (format_display_number(v), v < 0.0, false)
                                        }
                                        Ok(FormulaResult::Text(t)) => (t, false, false),
                                        Err(_) => ("ERR".to_string(), false, true),
                                    },
                                },
                                None => (String::new(), false, false),
                            };

                            // Truncate to fit column
                            let display: String = if text.len() > col_w {
                                text[..col_w - 1].to_string() + "~"
                            } else {
                                format!("{:<width$}", text, width = col_w)
                            };

                            let style = if is_cursor {
                                Style::default().bg(Color::Blue).fg(Color::White).add_modifier(Modifier::BOLD)
                            } else if is_err {
                                Style::default().fg(Color::Red).bg(row_bg)
                            } else if is_negative {
                                Style::default().fg(Color::Red).bg(row_bg)
                            } else {
                                Style::default().bg(row_bg)
                            };

                            spans.push(Span::styled(display, style));
                        }
                        lines.push(Line::from(spans));
                    }

                    let paragraph = Paragraph::new(lines);
                    f.render_widget(paragraph, inner);
                }

                pub fn move_cursor(&mut self, delta: (isize, isize)) {
                    self.cursor.col = max(0, self.cursor.col as isize + delta.0) as usize;
                    self.cursor.row = max(0, self.cursor.row as isize + delta.1) as usize;
                }

                /// Adjust scroll so cursor is visible within the given visible area.
                pub fn ensure_cursor_visible(&mut self, visible_cols: usize, visible_rows: usize) {
                    if visible_cols > 0 {
                        if self.cursor.col < self.scroll_col {
                            self.scroll_col = self.cursor.col;
                        } else if self.cursor.col >= self.scroll_col + visible_cols {
                            self.scroll_col = self.cursor.col - visible_cols + 1;
                        }
                    }
                    if visible_rows > 0 {
                        if self.cursor.row < self.scroll_row {
                            self.scroll_row = self.cursor.row;
                        } else if self.cursor.row >= self.scroll_row + visible_rows {
                            self.scroll_row = self.cursor.row - visible_rows + 1;
                        }
                    }
                }
            }

    }

    pub mod formula_bar {

            use std::cmp::min;

            use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
            use ratatui::{
                layout::Rect,
                style::{Modifier, Style},
                text::{Span, Line},
                widgets::{Block, Borders, Paragraph},
            };

            pub struct FormulaBar {
                pub text: String,
                pub cursor_pos: usize,
            }

            impl FormulaBar {
                pub fn new() -> Self {
                    FormulaBar {
                        text: String::new(),
                        cursor_pos: 0,
                    }
                }

                pub fn draw(&self, f: &mut ratatui::Frame, area: Rect, editing: bool, title_override: Option<&str>) {
                    let title = if let Some(t) = title_override {
                        t.to_string()
                    } else if editing {
                        " Edit (Enter=commit, Esc=cancel) ".to_string()
                    } else {
                        " Formula ".to_string()
                    };
                    let block = Block::default()
                        .borders(Borders::ALL)
                        .title(title);

                    if editing {
                        let mut spans = Vec::new();
                        for (i, ch) in self.text.chars().enumerate() {
                            let style = if i == self.cursor_pos {
                                Style::default().add_modifier(Modifier::REVERSED)
                            } else {
                                Style::default()
                            };
                            spans.push(Span::styled(ch.to_string(), style));
                        }
                        if self.cursor_pos == self.text.len() {
                            spans.push(Span::styled(
                                " ".to_string(),
                                Style::default().add_modifier(Modifier::REVERSED),
                            ));
                        }
                        let paragraph = Paragraph::new(Line::from(spans)).block(block);
                        f.render_widget(paragraph, area);
                    } else {
                        let paragraph = Paragraph::new(self.text.clone()).block(block);
                        f.render_widget(paragraph, area);
                    }
                }

                pub fn handle_input(&mut self, key: KeyEvent) -> Option<Action> {
                    match key.code {
                        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.text.insert(self.cursor_pos, c);
                            self.cursor_pos += 1;
                        }
                        KeyCode::Backspace => {
                            if self.cursor_pos > 0 && !self.text.is_empty() {
                                self.text.remove(self.cursor_pos - 1);
                                self.cursor_pos -= 1;
                            }
                        }
                        KeyCode::Delete => {
                            if self.cursor_pos < self.text.len() && !self.text.is_empty() {
                                self.text.remove(self.cursor_pos);
                            }
                        }
                        KeyCode::Left => {
                            if self.cursor_pos > 0 {
                                self.cursor_pos -= 1;
                            }
                        }
                        KeyCode::Right => {
                            if self.cursor_pos < self.text.len() {
                                self.cursor_pos += 1;
                            }
                        }
                        KeyCode::Home => self.cursor_pos = 0,
                        KeyCode::End => self.cursor_pos = self.text.len(),
                        KeyCode::Enter => return Some(Action::Commit(self.text.clone())),
                        KeyCode::Tab => return Some(Action::CommitTab(self.text.clone())),
                        KeyCode::Esc => return Some(Action::Cancel),
                        _ => {}
                    }
                    self.cursor_pos = min(self.cursor_pos, self.text.len());
                    None
                }
            }

            pub enum Action {
                Commit(String),
                CommitTab(String),
                Cancel,
            }

    }

    pub mod terminal {

            use std::io::{self, Stdout};

            use crossterm::{
                event::{read, Event, KeyCode, KeyModifiers},
                execute,
                terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
            };
            use ratatui::{
                backend::CrosstermBackend,
                layout::{Constraint, Direction, Layout},
                prelude::*,
                style::{Color, Style},
                text::{Line, Span},
                widgets::Paragraph,
            };
            use crate::core::address::{Address, format_address};
            use crate::core::cell::CellValue;
            use crate::command::action::Action as CmdAction;
            use crate::command::history::History;
            use crate::ui::formula_bar::{Action, FormulaBar};
            use crate::ui::grid_view::GridView;

            pub type RatatuiTerminal = Terminal<CrosstermBackend<Stdout>>;

            /// Input mode for the formula bar.
            #[derive(PartialEq)]
            enum InputMode {
                Normal,
                Editing,
                PromptSave,
                PromptLoad,
                ConfirmQuit,
            }

            pub struct TerminalApp {
                pub terminal: RatatuiTerminal,
                pub grid: GridView,
                pub formula_bar: FormulaBar,
            }

            impl TerminalApp {
                pub fn new(viewport: (usize, usize)) -> Result<Self, anyhow::Error> {
                    enable_raw_mode()?;
                    let mut stdout = io::stdout();
                    execute!(stdout, EnterAlternateScreen)?;
                    let backend = CrosstermBackend::new(stdout);
                    Ok(Self {
                        terminal: Terminal::new(backend)?,
                        grid: GridView::new(viewport),
                        formula_bar: FormulaBar::new(),
                    })
                }

                pub fn run(&mut self, app_state: &mut AppState) -> Result<(), anyhow::Error> {
                    let mut mode = InputMode::Normal;
                    let mut history = History::new();
                    let mut dirty = false;
                    let mut clipboard: Option<CellValue> = None;
                    let mut status_message: Option<String> = None;

                    loop {
                        // Compute visible area for scrolling
                        let term_area = self.terminal.size()?;
                        let grid_area_height = term_area.height.saturating_sub(4); // formula bar (3) + status (1)
                        let (visible_cols, visible_rows) = self.grid.visible_dimensions(
                            ratatui::layout::Rect::new(0, 0, term_area.width, grid_area_height)
                        );
                        self.grid.ensure_cursor_visible(visible_cols, visible_rows);

                        // In normal mode, show the raw cell content in the formula bar
                        if mode == InputMode::Normal {
                            let cell = app_state.sheet.get_cell(&self.grid.cursor);
                            self.formula_bar.text = match cell {
                                Some(c) => match &c.value {
                                    CellValue::Formula(s) => format!("={}", s),
                                    CellValue::Number(n) => n.to_string(),
                                    CellValue::Text(t) => t.clone(),
                                },
                                None => String::new(),
                            };
                        }

                        let grid = &self.grid;
                        let formula_bar = &self.formula_bar;
                        let sheet = &app_state.sheet;
                        let current_mode = &mode;
                        let cursor_addr = format_address(&self.grid.cursor);
                        let status_msg = status_message.clone();

                        // Build formula bar title
                        let fb_title: Option<String> = match current_mode {
                            InputMode::Editing => {
                                Some(format!(" Edit [{}]: {} ", cursor_addr, formula_bar.text))
                            }
                            InputMode::PromptSave => Some(" Save CSV path (Enter=confirm, Esc=cancel): ".to_string()),
                            InputMode::PromptLoad => Some(" Load CSV path (Enter=confirm, Esc=cancel): ".to_string()),
                            _ => None,
                        };
                        let is_editing = mode != InputMode::Normal && mode != InputMode::ConfirmQuit;

                        // Status bar help text
                        let help_text = match current_mode {
                            InputMode::Normal => {
                                "Arrows:move  Enter:edit  q:quit  Ctrl+S:save  Ctrl+O:open  Ctrl+Z:undo  Ctrl+Y:redo  Ctrl+C:copy  Ctrl+V:paste  +/-:col width"
                            }
                            InputMode::Editing | InputMode::PromptSave | InputMode::PromptLoad => {
                                "Enter:commit  Esc:cancel  Tab:commit+right  Ctrl+C:quit"
                            }
                            InputMode::ConfirmQuit => {
                                "Unsaved changes! y:quit  n:cancel"
                            }
                        };

                        self.terminal.draw(|f| {
                            let area = f.area();
                            let chunks = Layout::default()
                                .direction(Direction::Vertical)
                                .constraints([
                                    Constraint::Min(5),
                                    Constraint::Length(3),
                                    Constraint::Length(1),
                                ])
                                .split(area);

                            grid.draw(f, sheet, chunks[0]);
                            formula_bar.draw(f, chunks[1], is_editing, fb_title.as_deref());

                            // Status bar
                            let status_text = if let Some(ref msg) = status_msg {
                                msg.clone()
                            } else {
                                help_text.to_string()
                            };
                            let status_line = Line::from(vec![
                                Span::styled(
                                    format!(" {} ", status_text),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]);
                            let status_bar = Paragraph::new(status_line);
                            f.render_widget(status_bar, chunks[2]);
                        })?;

                        // Clear one-shot status messages
                        status_message = None;

                        match read()? {
                            Event::Key(key) => {
                                match mode {
                                    InputMode::ConfirmQuit => {
                                        match key.code {
                                            KeyCode::Char('y') | KeyCode::Char('Y') => break,
                                            _ => { mode = InputMode::Normal; }
                                        }
                                    }
                                    InputMode::Editing => {
                                        // Ctrl+C quits from editing mode
                                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                                            break;
                                        }
                                        let action = self.formula_bar.handle_input(key);
                                        let was_tab = matches!(&action, Some(Action::CommitTab(_)));
                                        match action {
                                            Some(Action::Commit(text)) | Some(Action::CommitTab(text)) => {
                                                let addr = self.grid.cursor;
                                                // Save old value for undo
                                                let old_value = app_state.sheet.get_cell(&addr)
                                                    .map(|c| c.value.clone());

                                                let value = if text.starts_with('=') {
                                                    CellValue::Formula(text[1..].to_string())
                                                } else if let Ok(n) = text.parse::<f64>() {
                                                    CellValue::Number(n)
                                                } else {
                                                    CellValue::Text(text)
                                                };
                                                let _ = app_state.sheet.set_cell_value(addr, value);

                                                // Push undo action
                                                if let Some(old) = old_value {
                                                    history.push(CmdAction::SetCell(addr, old));
                                                } else {
                                                    history.push(CmdAction::ClearCell(addr, CellValue::Number(0.0)));
                                                }

                                                dirty = true;
                                                mode = InputMode::Normal;

                                                // Move cursor after commit
                                                if was_tab {
                                                    self.grid.move_cursor((1, 0));
                                                } else {
                                                    self.grid.move_cursor((0, 1));
                                                }
                                            }
                                            Some(Action::Cancel) => {
                                                mode = InputMode::Normal;
                                            }
                                            None => {}
                                        }
                                    }
                                    InputMode::PromptSave => {
                                        match self.formula_bar.handle_input(key) {
                                            Some(Action::Commit(path_str)) | Some(Action::CommitTab(path_str)) => {
                                                if !path_str.is_empty() {
                                                    let p = std::path::Path::new(&path_str);
                                                    match crate::core::csv::export_csv(p, &app_state.sheet) {
                                                        Ok(_) => {
                                                            app_state.file_path = Some(path_str);
                                                            dirty = false;
                                                            status_message = Some("Saved.".to_string());
                                                        }
                                                        Err(e) => {
                                                            status_message = Some(format!("Save error: {:?}", e));
                                                        }
                                                    }
                                                }
                                                mode = InputMode::Normal;
                                            }
                                            Some(Action::Cancel) => { mode = InputMode::Normal; }
                                            None => {}
                                        }
                                    }
                                    InputMode::PromptLoad => {
                                        match self.formula_bar.handle_input(key) {
                                            Some(Action::Commit(path_str)) | Some(Action::CommitTab(path_str)) => {
                                                if !path_str.is_empty() {
                                                    let p = std::path::Path::new(&path_str);
                                                    match crate::core::csv::import_csv_to_sheet(p, &mut app_state.sheet) {
                                                        Ok(_) => {
                                                            app_state.file_path = Some(path_str);
                                                            dirty = false;
                                                            status_message = Some("Loaded.".to_string());
                                                        }
                                                        Err(e) => {
                                                            status_message = Some(format!("Load error: {:?}", e));
                                                        }
                                                    }
                                                }
                                                mode = InputMode::Normal;
                                            }
                                            Some(Action::Cancel) => { mode = InputMode::Normal; }
                                            None => {}
                                        }
                                    }
                                    InputMode::Normal => {
                                        // Ctrl key combos
                                        if key.modifiers.contains(KeyModifiers::CONTROL) {
                                            match key.code {
                                                KeyCode::Char('c') => {
                                                    // Copy current cell
                                                    clipboard = app_state.sheet.get_cell(&self.grid.cursor)
                                                        .map(|c| c.value.clone());
                                                }
                                                KeyCode::Char('z') => {
                                                    // Undo
                                                    if let Some(action) = history.undo() {
                                                        match action {
                                                            CmdAction::SetCell(addr, old_val) => {
                                                                // Save current value for redo
                                                                let current = app_state.sheet.get_cell(&addr)
                                                                    .map(|c| c.value.clone());
                                                                let _ = app_state.sheet.set_cell_value(addr, old_val);
                                                                if let Some(cur) = current {
                                                                    history.push_redo(CmdAction::SetCell(addr, cur));
                                                                }
                                                                dirty = true;
                                                            }
                                                            CmdAction::ClearCell(addr, _) => {
                                                                let current = app_state.sheet.get_cell(&addr)
                                                                    .map(|c| c.value.clone());
                                                                app_state.sheet.clear_cell(&addr);
                                                                if let Some(cur) = current {
                                                                    history.push_redo(CmdAction::SetCell(addr, cur));
                                                                } else {
                                                                    history.push_redo(CmdAction::ClearCell(addr, CellValue::Number(0.0)));
                                                                }
                                                                dirty = true;
                                                            }
                                                        }
                                                    }
                                                }
                                                KeyCode::Char('y') => {
                                                    // Redo
                                                    if let Some(action) = history.redo() {
                                                        match action {
                                                            CmdAction::SetCell(addr, val) => {
                                                                let current = app_state.sheet.get_cell(&addr)
                                                                    .map(|c| c.value.clone());
                                                                let _ = app_state.sheet.set_cell_value(addr, val);
                                                                if let Some(cur) = current {
                                                                    history.push_undo(CmdAction::SetCell(addr, cur));
                                                                } else {
                                                                    history.push_undo(CmdAction::ClearCell(addr, CellValue::Number(0.0)));
                                                                }
                                                                dirty = true;
                                                            }
                                                            CmdAction::ClearCell(addr, _) => {
                                                                let current = app_state.sheet.get_cell(&addr)
                                                                    .map(|c| c.value.clone());
                                                                app_state.sheet.clear_cell(&addr);
                                                                if let Some(cur) = current {
                                                                    history.push_undo(CmdAction::SetCell(addr, cur));
                                                                } else {
                                                                    history.push_undo(CmdAction::ClearCell(addr, CellValue::Number(0.0)));
                                                                }
                                                                dirty = true;
                                                            }
                                                        }
                                                    }
                                                }
                                                KeyCode::Char('s') => {
                                                    // Save
                                                    if let Some(ref path) = app_state.file_path {
                                                        let p = std::path::Path::new(path);
                                                        match crate::core::csv::export_csv(p, &app_state.sheet) {
                                                            Ok(_) => {
                                                                dirty = false;
                                                                status_message = Some("Saved.".to_string());
                                                            }
                                                            Err(e) => {
                                                                status_message = Some(format!("Save error: {:?}", e));
                                                            }
                                                        }
                                                    } else {
                                                        self.formula_bar.text.clear();
                                                        self.formula_bar.cursor_pos = 0;
                                                        mode = InputMode::PromptSave;
                                                    }
                                                }
                                                KeyCode::Char('o') => {
                                                    // Open/Load
                                                    self.formula_bar.text.clear();
                                                    self.formula_bar.cursor_pos = 0;
                                                    mode = InputMode::PromptLoad;
                                                }
                                                KeyCode::Char('v') => {
                                                    // Paste
                                                    if let Some(ref val) = clipboard {
                                                        let addr = self.grid.cursor;
                                                        let old_value = app_state.sheet.get_cell(&addr)
                                                            .map(|c| c.value.clone());
                                                        let _ = app_state.sheet.set_cell_value(addr, val.clone());
                                                        if let Some(old) = old_value {
                                                            history.push(CmdAction::SetCell(addr, old));
                                                        } else {
                                                            history.push(CmdAction::ClearCell(addr, CellValue::Number(0.0)));
                                                        }
                                                        dirty = true;
                                                    }
                                                }
                                                KeyCode::Home => {
                                                    // Ctrl+Home: jump to A1
                                                    self.grid.cursor = Address { row: 0, col: 0 };
                                                }
                                                _ => {}
                                            }
                                            continue;
                                        }

                                        // Shift+Tab
                                        if key.modifiers.contains(KeyModifiers::SHIFT) && key.code == KeyCode::BackTab {
                                            self.grid.move_cursor((-1, 0));
                                            continue;
                                        }

                                        match key.code {
                                            KeyCode::Char('q') => {
                                                if dirty {
                                                    mode = InputMode::ConfirmQuit;
                                                } else {
                                                    break;
                                                }
                                            }
                                            KeyCode::Enter | KeyCode::F(2) => {
                                                // Edit current cell
                                                let cell = app_state.sheet.get_cell(&self.grid.cursor);
                                                if let Some(cell) = cell {
                                                    match &cell.value {
                                                        CellValue::Formula(s) => {
                                                            self.formula_bar.text = format!("={}", s);
                                                        }
                                                        CellValue::Number(n) => {
                                                            self.formula_bar.text = n.to_string();
                                                        }
                                                        CellValue::Text(t) => {
                                                            self.formula_bar.text = t.clone();
                                                        }
                                                    }
                                                } else {
                                                    self.formula_bar.text.clear();
                                                }
                                                self.formula_bar.cursor_pos = self.formula_bar.text.len();
                                                mode = InputMode::Editing;
                                            }
                                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                                // Increase column width
                                                if self.grid.col_width < 30 {
                                                    self.grid.col_width += 2;
                                                }
                                            }
                                            KeyCode::Char('-') => {
                                                // Decrease column width
                                                if self.grid.col_width > 4 {
                                                    self.grid.col_width -= 2;
                                                }
                                            }
                                            KeyCode::Char(c) => {
                                                // Start editing with the typed character
                                                self.formula_bar.text = c.to_string();
                                                self.formula_bar.cursor_pos = 1;
                                                mode = InputMode::Editing;
                                            }
                                            KeyCode::Delete | KeyCode::Backspace => {
                                                let addr = self.grid.cursor;
                                                let old_value = app_state.sheet.get_cell(&addr)
                                                    .map(|c| c.value.clone());
                                                app_state.sheet.clear_cell(&addr);
                                                if let Some(old) = old_value {
                                                    history.push(CmdAction::SetCell(addr, old));
                                                    dirty = true;
                                                }
                                            }
                                            KeyCode::Left => self.grid.move_cursor((-1, 0)),
                                            KeyCode::Right => self.grid.move_cursor((1, 0)),
                                            KeyCode::Up => self.grid.move_cursor((0, -1)),
                                            KeyCode::Down => self.grid.move_cursor((0, 1)),
                                            KeyCode::Tab => self.grid.move_cursor((1, 0)),
                                            KeyCode::PageUp => {
                                                self.grid.move_cursor((0, -(visible_rows as isize)));
                                            }
                                            KeyCode::PageDown => {
                                                self.grid.move_cursor((0, visible_rows as isize));
                                            }
                                            KeyCode::Home => {
                                                self.grid.cursor.col = 0;
                                            }
                                            KeyCode::End => {
                                                if let Some(max_col) = app_state.sheet.max_occupied_col_in_row(self.grid.cursor.row) {
                                                    self.grid.cursor.col = max_col;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    disable_raw_mode()?;
                    execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
                    self.terminal.show_cursor()?;

                    Ok(())
                }
            }

            pub struct AppState {
                pub sheet: crate::core::sheet::Sheet,
                pub engine: crate::core::formula_engine::SimpleFormulaEngine,
                pub file_path: Option<String>,
            }

    }

}

fn main() {
    // Phase 4b: CLI argument for file path
    let file_path = std::env::args().nth(1);

    let mut app_state = app::AppState::new();
    if let Err(e) = app::main_loop(&mut app_state, file_path) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
