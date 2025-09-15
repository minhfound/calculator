# fx-880 Emulator (ClassWiz-like)

A Python application that emulates basic features of the Casio fx-880 / ClassWiz calculators.  
Built with **Tkinter**, it provides a graphical interface with multiple modes for calculation, statistics, probability distributions, spreadsheets, value tables, and polynomial equation solving.

---

## ✨ Features

- **Calculate**
  - Enter and evaluate mathematical expressions.
  - Supports most functions from Python’s `math` module (`sin`, `cos`, `log`, `sqrt`, etc.).
  - Uses a safe AST-based evaluator (no raw `eval`).

- **Statistics**
  - Enter datasets (comma, space, or newline separated).
  - Computes:
    - Count (`n`)
    - Mean
    - Median
    - Population and sample standard deviation
  - If input is pairs `(x, y)`, performs **linear regression** (`y = ax + b`).

- **Distribution**
  - **Normal distribution**: PDF & CDF.
  - **Binomial distribution**: PMF & CDF.

- **Spreadsheet**
  - Simple grid (15 rows × 8 columns).
  - Import/export CSV files.
  - Clear/reset table.

- **Table**
  - Build function value tables for expressions with variable `x`.
  - Define range `[a..b]` and step size.

- **Equation**
  - Solve polynomial equations from coefficients.
  - Uses:
    - `numpy.roots` (if NumPy installed),
    - or `sympy.nroots` (if SymPy installed),
    - or a basic fallback numeric solver.

---

## 📦 Requirements

- **Python 3.8+**
- Optional (for better equation solving):
  - [NumPy](https://numpy.org/)
  - [SymPy](https://www.sympy.org/)

---

## 🚀 Run

Clone the repo and run:

```bash
python fx880_emulator.py
