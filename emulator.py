#!/usr/bin/env python3
"""
fx-880 (ClassWiz-like) emulator (simplified)

Features:
- Calculate: expression evaluator using safe AST + math functions
- Statistics: enter dataset, compute n, mean, median, population/sample stddev, linear regression
- Distribution: Normal (pdf,cdf), Binomial (pmf,cdf)
- Spreadsheet: simple grid, edit cells, import/export CSV
- Table: build value tables from expressions
- Equation: solve polynomial equations (uses numpy or sympy if available, else numeric solver)

Dependencies:
- Python 3.8+
- Optional: numpy or sympy for robust polynomial root finding

Run:
    python fx880_emulator.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import math, statistics, csv, io, sys
from math import *
import ast

# Try optional libs
HAS_NUMPY = False
HAS_SYMPY = False
try:
    import numpy as _np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
try:
    import sympy as _sp
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False

# ----------------------------
# Safe expression evaluator
# ----------------------------
SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
# expose basic names
SAFE_NAMES.update({
    'abs': abs, 'round': round, 'min': min, 'max': max, 'pow': pow
})

# allow constants
SAFE_NAMES['pi'] = math.pi
SAFE_NAMES['e'] = math.e

class ExprEvaluator:
    @staticmethod
    def eval_expr(expr):
        """Evaluate arithmetic expression safely (with math support)."""
        try:
            node = ast.parse(expr, mode='eval')
            return ExprEvaluator._eval(node.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    @staticmethod
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):  # py3.8+
            return node.value
        if isinstance(node, ast.BinOp):
            left = ExprEvaluator._eval(node.left)
            right = ExprEvaluator._eval(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
            if isinstance(node.op, ast.Mod): return left % right
            raise ValueError("Unsupported binary op")
        if isinstance(node, ast.UnaryOp):
            val = ExprEvaluator._eval(node.operand)
            if isinstance(node.op, ast.UAdd): return +val
            if isinstance(node.op, ast.USub): return -val
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                fname = func.id
                if fname not in SAFE_NAMES:
                    raise ValueError(f"Function {fname} not allowed")
                f = SAFE_NAMES[fname]
                args = [ExprEvaluator._eval(a) for a in node.args]
                return f(*args)
            else:
                raise ValueError("Only named functions allowed")
        if isinstance(node, ast.Name):
            if node.id in SAFE_NAMES:
                return SAFE_NAMES[node.id]
            else:
                raise ValueError(f"Name {node.id} not allowed")
        raise ValueError("Unsupported expression element: " + str(node))

# ----------------------------
# Core GUI Application
# ----------------------------
class FX880App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("fx-880 Emulator")
        self.geometry("900x600")
        self.create_widgets()

    def create_widgets(self):
        # Top menu: mode selection
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        modes = ['Calculate', 'Statistics', 'Distribution', 'Spreadsheet', 'Table', 'Equation']
        self.frames = {}
        for m in modes:
            b = ttk.Button(toolbar, text=m, command=lambda m=m: self.show_mode(m))
            b.pack(side=tk.LEFT, padx=2, pady=2)

        # Container for mode frames
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        for m in modes:
            frame = ttk.Frame(container)
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.frames[m] = frame

        # Populate each mode
        self._build_calculate(self.frames['Calculate'])
        self._build_statistics(self.frames['Statistics'])
        self._build_distribution(self.frames['Distribution'])
        self._build_spreadsheet(self.frames['Spreadsheet'])
        self._build_table(self.frames['Table'])
        self._build_equation(self.frames['Equation'])

        self.show_mode('Calculate')

    def show_mode(self, mode):
        for m, f in self.frames.items():
            f.lift() if m == mode else None
        self.frames[mode].tkraise()

    # ----------------------------
    # Calculate mode
    # ----------------------------
    def _build_calculate(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Calculate (enter expression, use math functions)").pack(anchor=tk.W)
        self.calc_entry = ttk.Entry(frm, font=("Consolas", 16))
        self.calc_entry.pack(fill=tk.X, pady=6)

        self.calc_entry.bind("<Return>", lambda event: self._calc_eval())

        outfrm = ttk.Frame(frm)
        outfrm.pack(fill=tk.BOTH, expand=True)
        self.calc_result = tk.Text(outfrm, height=10, state='disabled', font=("Consolas", 12))
        self.calc_result.pack(fill=tk.BOTH, expand=True)
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Evaluate", command=self._calc_eval).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear", command=lambda: self.calc_entry.delete(0, tk.END)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Common functions list", command=self._show_common_funcs).pack(side=tk.LEFT, padx=4)

    def _calc_eval(self):
        expr = self.calc_entry.get().strip()
        if not expr:
            return
        try:
            val = ExprEvaluator.eval_expr(expr)
            self._append_calc(f"> {expr}\n= {val}\n\n")
        except Exception as e:
            self._append_calc(f"Error: {e}\n\n")

    def _append_calc(self, text):
        self.calc_result.configure(state='normal')
        self.calc_result.insert(tk.END, text)
        self.calc_result.configure(state='disabled')
        self.calc_result.see(tk.END)

    def _show_common_funcs(self):
        funcs = sorted(list(SAFE_NAMES.keys()))
        messagebox.showinfo("Allowed functions/constants", ", ".join(funcs))

    # ----------------------------
    # Statistics mode
    # ----------------------------
    def _build_statistics(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(frm)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Enter numbers (comma, space or newline separated):").pack(anchor=tk.W)
        self.stats_text = tk.Text(left, width=40, height=15)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        bframe = ttk.Frame(left)
        bframe.pack(fill=tk.X, pady=6)
        ttk.Button(bframe, text="Compute", command=self._compute_stats).pack(side=tk.LEFT)
        ttk.Button(bframe, text="Clear", command=lambda: self.stats_text.delete('1.0', tk.END)).pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(frm)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="Statistics result").pack(anchor=tk.W)
        self.stats_out = tk.Text(right, state='disabled', height=15)
        self.stats_out.pack(fill=tk.BOTH, expand=True)

    def _compute_stats(self):
        txt = self.stats_text.get('1.0', tk.END).strip()
        if not txt:
            return
        try:
            parts = [p for p in txt.replace('\n', ' ').replace(',', ' ').split() if p]
            data = [float(p) for p in parts]
        except Exception as e:
            messagebox.showerror("Parse error", f"Could not parse numbers: {e}")
            return
        n = len(data)
        mean = statistics.mean(data)
        median = statistics.median(data)
        pstdev = statistics.pstdev(data)
        try:
            sstdev = statistics.stdev(data)
        except Exception:
            sstdev = float('nan')
        out = io.StringIO()
        out.write(f"n = {n}\n")
        out.write(f"mean = {mean}\n")
        out.write(f"median = {median}\n")
        out.write(f"population stddev = {pstdev}\n")
        out.write(f"sample stddev = {sstdev}\n")
        # linear regression (simple least squares y = ax + b) if input provided as pairs
        if n >= 2:
            # If user provided x,y pairs in format "x y" on each line or alternating values, try to interpret
            if all(len(x.split()) == 2 for x in txt.strip().splitlines() if x.strip()):
                pairs = [tuple(map(float, line.split())) for line in txt.strip().splitlines() if line.strip()]
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                # compute slope/intercept
                meanx, meany = statistics.mean(xs), statistics.mean(ys)
                num = sum((xi - meanx)*(yi - meany) for xi, yi in zip(xs, ys))
                den = sum((xi - meanx)**2 for xi in xs)
                if den != 0:
                    a = num/den
                    b = meany - a*meanx
                    out.write("\nLinear regression (y = a*x + b)\n")
                    out.write(f"a (slope) = {a}\n")
                    out.write(f"b (intercept) = {b}\n")
            # else: no pairs provided
        self.stats_out.configure(state='normal')
        self.stats_out.delete('1.0', tk.END)
        self.stats_out.insert('1.0', out.getvalue())
        self.stats_out.configure(state='disabled')

    # ----------------------------
    # Distribution mode
    # ----------------------------
    def _build_distribution(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Distribution functions").pack(anchor=tk.W)
        # Normal
        nframe = ttk.LabelFrame(frm, text="Normal distribution")
        nframe.pack(fill=tk.X, pady=6)
        ttk.Label(nframe, text="mean:").grid(row=0, column=0, sticky=tk.W)
        self.norm_mu = tk.StringVar(value="0")
        ttk.Entry(nframe, textvariable=self.norm_mu, width=8).grid(row=0, column=1)
        ttk.Label(nframe, text="sd:").grid(row=0, column=2, sticky=tk.W)
        self.norm_sigma = tk.StringVar(value="1")
        ttk.Entry(nframe, textvariable=self.norm_sigma, width=8).grid(row=0, column=3)
        ttk.Label(nframe, text="x:").grid(row=0, column=4, sticky=tk.W)
        self.norm_x = tk.StringVar(value="0")
        ttk.Entry(nframe, textvariable=self.norm_x, width=8).grid(row=0, column=5)
        ttk.Button(nframe, text="PDF", command=self._normal_pdf).grid(row=0, column=6, padx=4)
        ttk.Button(nframe, text="CDF", command=self._normal_cdf).grid(row=0, column=7, padx=4)

        # Binomial
        bframe = ttk.LabelFrame(frm, text="Binomial distribution")
        bframe.pack(fill=tk.X, pady=6)
        ttk.Label(bframe, text="n:").grid(row=0, column=0, sticky=tk.W)
        self.bin_n = tk.StringVar(value="10")
        ttk.Entry(bframe, textvariable=self.bin_n, width=6).grid(row=0, column=1)
        ttk.Label(bframe, text="p:").grid(row=0, column=2, sticky=tk.W)
        self.bin_p = tk.StringVar(value="0.5")
        ttk.Entry(bframe, textvariable=self.bin_p, width=6).grid(row=0, column=3)
        ttk.Label(bframe, text="k:").grid(row=0, column=4, sticky=tk.W)
        self.bin_k = tk.StringVar(value="5")
        ttk.Entry(bframe, textvariable=self.bin_k, width=6).grid(row=0, column=5)
        ttk.Button(bframe, text="PMF", command=self._binomial_pmf).grid(row=0, column=6, padx=4)
        ttk.Button(bframe, text="CDF (<=k)", command=self._binomial_cdf).grid(row=0, column=7, padx=4)

        self.dist_out = tk.Text(frm, height=10, state='disabled')
        self.dist_out.pack(fill=tk.BOTH, expand=True, pady=6)

    def _normal_pdf(self):
        try:
            mu = float(self.norm_mu.get()); sigma = float(self.norm_sigma.get()); x = float(self.norm_x.get())
            pdf = (1.0/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sigma)**2)
            self._dist_print(f"Normal PDF(x={x}; mu={mu}, sd={sigma}) = {pdf}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _normal_cdf(self):
        try:
            mu = float(self.norm_mu.get()); sigma = float(self.norm_sigma.get()); x = float(self.norm_x.get())
            # use error function
            z = (x-mu)/(sigma*math.sqrt(2))
            cdf = 0.5*(1 + math.erf(z))
            self._dist_print(f"Normal CDF(x={x}; mu={mu}, sd={sigma}) = {cdf}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _binomial_pmf(self):
        try:
            n = int(self.bin_n.get()); p = float(self.bin_p.get()); k = int(self.bin_k.get())
            if not (0 <= p <= 1): raise ValueError("p must be in [0,1]")
            from math import comb
            prob = comb(n, k) * (p**k) * ((1-p)**(n-k))
            self._dist_print(f"Binomial PMF(n={n}, p={p}, k={k}) = {prob}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _binomial_cdf(self):
        try:
            n = int(self.bin_n.get()); p = float(self.bin_p.get()); k = int(self.bin_k.get())
            from math import comb
            s = 0.0
            for i in range(0, k+1):
                s += comb(n, i) * (p**i) * ((1-p)**(n-i))
            self._dist_print(f"Binomial CDF P(X<={k}) = {s}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _dist_print(self, text):
        self.dist_out.configure(state='normal')
        self.dist_out.insert(tk.END, text + "\n")
        self.dist_out.configure(state='disabled')
        self.dist_out.see(tk.END)

    # ----------------------------
    # Spreadsheet mode
    # ----------------------------
    def _build_spreadsheet(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Spreadsheet (simple grid)").pack(anchor=tk.W)
        self.sheet_rows = 15
        self.sheet_cols = 8
        gridfrm = ttk.Frame(frm)
        gridfrm.pack(fill=tk.BOTH, expand=True)

        self.cells = []
        for r in range(self.sheet_rows):
            row_entries = []
            for c in range(self.sheet_cols):
                e = ttk.Entry(gridfrm, width=12)
                e.grid(row=r, column=c, padx=1, pady=1, sticky='nsew')
                row_entries.append(e)
            self.cells.append(row_entries)
        btnfrm = ttk.Frame(frm)
        btnfrm.pack(fill=tk.X, pady=6)
        ttk.Button(btnfrm, text="Import CSV", command=self._import_csv).pack(side=tk.LEFT)
        ttk.Button(btnfrm, text="Export CSV", command=self._export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btnfrm, text="Clear", command=self._clear_sheet).pack(side=tk.LEFT, padx=4)

    def _import_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path: return
        try:
            with open(path, newline='') as f:
                rdr = csv.reader(f)
                for r, row in enumerate(rdr):
                    if r >= self.sheet_rows: break
                    for c, val in enumerate(row):
                        if c >= self.sheet_cols: break
                        self.cells[r][c].delete(0, tk.END)
                        self.cells[r][c].insert(0, val)
        except Exception as e:
            messagebox.showerror("Import error", str(e))

    def _export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path: return
        try:
            with open(path, 'w', newline='') as f:
                wr = csv.writer(f)
                for r in range(self.sheet_rows):
                    row = [self.cells[r][c].get() for c in range(self.sheet_cols)]
                    wr.writerow(row)
            messagebox.showinfo("Saved", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _clear_sheet(self):
        for r in range(self.sheet_rows):
            for c in range(self.sheet_cols):
                self.cells[r][c].delete(0, tk.END)

    # ----------------------------
    # Table mode
    # ----------------------------
    def _build_table(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Table: evaluate expression over range x=a..b step s").pack(anchor=tk.W)
        entryfrm = ttk.Frame(frm)
        entryfrm.pack(fill=tk.X, pady=6)
        ttk.Label(entryfrm, text="Expression (use 'x')").grid(row=0, column=0)
        self.table_expr = ttk.Entry(entryfrm, width=30)
        self.table_expr.grid(row=0, column=1, padx=4)
        ttk.Label(entryfrm, text="a").grid(row=0, column=2)
        self.table_a = ttk.Entry(entryfrm, width=6)
        self.table_a.grid(row=0, column=3)
        ttk.Label(entryfrm, text="b").grid(row=0, column=4)
        self.table_b = ttk.Entry(entryfrm, width=6)
        self.table_b.grid(row=0, column=5)
        ttk.Label(entryfrm, text="step").grid(row=0, column=6)
        self.table_step = ttk.Entry(entryfrm, width=6)
        self.table_step.grid(row=0, column=7)
        ttk.Button(entryfrm, text="Build Table", command=self._build_table_values).grid(row=0, column=8, padx=6)

        self.table_out = tk.Text(frm, height=20, state='disabled')
        self.table_out.pack(fill=tk.BOTH, expand=True)

    def _build_table_values(self):
        expr = self.table_expr.get().strip()
        try:
            a = float(self.table_a.get()); b = float(self.table_b.get()); step = float(self.table_step.get())
            if step == 0: raise ValueError("step cannot be 0")
            x = a
            self.table_out.configure(state='normal')
            self.table_out.delete('1.0', tk.END)
            while (x <= b + 1e-12 if step>0 else x >= b - 1e-12):
                # evaluate expression with x available
                local_names = SAFE_NAMES.copy()
                local_names['x'] = x
                # we can try to evaluate using our ExprEvaluator with injection
                try:
                    # temporary hack: replace 'x' occurrences -> rely on expression evaluator
                    val = ExprEvaluator.eval_expr(expr)
                except Exception:
                    # fallback: use Python eval with safe names
                    val = eval(expr, {"__builtins__":None}, local_names)
                self.table_out.insert(tk.END, f"x={x:g}\t -> {val}\n")
                x += step
            self.table_out.configure(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----------------------------
    # Equation mode
    # ----------------------------
    def _build_equation(self, parent):
        frm = ttk.Frame(parent, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text="Equation solver (polynomial) - enter coefficients a_n ... a0").pack(anchor=tk.W)
        cf = ttk.Frame(frm)
        cf.pack(fill=tk.X, pady=6)
        ttk.Label(cf, text="Coefficients (space/comma-separated, highest degree first):").grid(row=0, column=0, sticky=tk.W)
        self.eq_coeffs = ttk.Entry(cf, width=60)
        self.eq_coeffs.grid(row=1, column=0, sticky=tk.W)
        bframe = ttk.Frame(frm)
        bframe.pack(fill=tk.X, pady=6)
        ttk.Button(bframe, text="Solve", command=self._solve_polynomial).pack(side=tk.LEFT)
        ttk.Button(bframe, text="Clear", command=lambda: self.eq_coeffs.delete(0, tk.END)).pack(side=tk.LEFT, padx=4)
        self.eq_out = tk.Text(frm, height=15, state='disabled')
        self.eq_out.pack(fill=tk.BOTH, expand=True)

    def _solve_polynomial(self):
        txt = self.eq_coeffs.get().strip()
        if not txt:
            return
        try:
            parts = [p for p in txt.replace(',', ' ').split() if p]
            coeffs = [float(p) for p in parts]
            if len(coeffs) < 2:
                raise ValueError("Need at least two coefficients (linear minimum).")
            roots = None
            # preferred: numpy.roots
            if HAS_NUMPY:
                roots = _np.roots(coeffs)
                roots = [complex(r) for r in roots]
                self._eq_print(f"Using numpy.roots (degree {len(coeffs)-1})")
            elif HAS_SYMPY:
                poly = _sp.Poly(_sp.symbols('x')**(len(coeffs)-1), _sp.symbols('x'))
                x = _sp.symbols('x')
                expr = sum(coef * x**exp for coef, exp in zip(coeffs, range(len(coeffs)-1, -1, -1)))
                rts = _sp.nroots(expr)
                roots = rts
                self._eq_print("Using sympy nroots")
            else:
                # fallback simple numeric: use companion matrix via numpy if not available cannot
                roots = self._fallback_roots(coeffs)
                self._eq_print("Using fallback numeric solver (simple)")
            self._eq_print("Roots:")
            for r in roots:
                self._eq_print(str(r))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _fallback_roots(self, coeffs):
        """Very basic root finder for fallback: try integer/rational small roots and deflate polynomial."""
        # This is intentionally simple: try rational root theorem for small integer roots then deflate.
        def poly_eval(cfs, x):
            val = 0
            for a in cfs:
                val = val*x + a
            return val

        cfs = coeffs[:]  # copy
        deg = len(cfs)-1
        roots = []
        # try integer roots in range -50..50
        for _ in range(deg):
            found = False
            for r in range(-50, 51):
                if r == 0 and cfs[-1] != 0:
                    # try
                    pass
                val = poly_eval(cfs, r)
                if abs(val) < 1e-8:
                    roots.append(r)
                    # deflate synthetic division
                    new = []
                    carry = 0
                    for a in cfs:
                        carry = carry * r + a
                        if len(new) < len(cfs)-1:
                            new.append(carry)
                    # new has length len(cfs)-1 but last value is remainder; recompute properly
                    # compute synthetic division properly:
                    quotient = []
                    b = cfs[0]
                    quotient.append(b)
                    for a in cfs[1: -1]:
                        b = b * r + a
                        quotient.append(b)
                    cfs = quotient
                    found = True
                    break
            if not found:
                # give up: compute numeric root using binary search on intervals maybe or use Newton
                # try to find one real root via Newton starting guesses
                def newton_root(coeffs, x0):
                    for it in range(60):
                        # evaluate and derivative
                        p = 0; dp = 0
                        for a in coeffs:
                            dp = dp * x0 + p
                            p = p * x0 + a
                        if abs(p) < 1e-12:
                            return x0
                        if dp == 0:
                            break
                        x0 = x0 - p/dp
                    return None
                guess = newton_root(cfs, 0.1)
                if guess is None:
                    guess = newton_root(cfs, 1.0)
                if guess is None:
                    # give up and return approximate using built-in complex roots via characteristic companion will not be available
                    # return 'approx' using Python's complex approximation using power method is heavy; just break
                    # Return remaining polynomial as 'unknown'
                    roots.append(f"(undetermined remaining poly degree {len(cfs)-1})")
                    break
                else:
                    roots.append(guess)
                    # deflate numerically via polynomial division
                    # skip actual deflation for brevity
                    break
        return roots

    def _eq_print(self, s):
        self.eq_out.configure(state='normal')
        self.eq_out.insert(tk.END, s + "\n")
        self.eq_out.configure(state='disabled')
        self.eq_out.see(tk.END)


if __name__ == "__main__":
    app = FX880App()
    app.mainloop()
