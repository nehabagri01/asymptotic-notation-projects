"""
app.py - Asymptotic Notation Visualizer (Single-file Streamlit app)

Requirements (for deployment):
    streamlit
    numpy
    matplotlib

Example requirements.txt contents (if you want one):
    streamlit
    numpy
    matplotlib

Run locally:
    pip install streamlit numpy matplotlib
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from io import BytesIO
import textwrap

# -------------------------
# Page configuration
# -------------------------
st.set_page_config(
    page_title="Asymptotic Notations Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Asymptotic Notation Visualizer — Big O, Ω, Θ")
st.caption("Interactive DAA mini-project — visualize, explain and compute asymptotic growth rates")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

n_max = st.sidebar.slider(
    "Maximum input size n (range will be 1..n)",
    min_value=10,
    max_value=2000,
    value=200,
    step=10
)

use_log_scale = st.sidebar.checkbox("Use logarithmic y-axis (recommended for large ranges)", value=False)

show_functions = st.sidebar.multiselect(
    "Select complexity functions to show",
    options=[
        "O(1)",
        "O(log n)",
        "O(n)",
        "O(n log n)",
        "O(n^2)",
        "O(2^n)"
    ],
    default=["O(1)", "O(n)", "O(n^2)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Adjust visual options:")
line_width = st.sidebar.slider("Line width", 1, 6, 2)
point_marker = st.sidebar.selectbox("Marker", options=["", "o", "s", "v", "^"], index=0)

# -------------------------
# Prepare n and functions
# -------------------------
# Use numpy array; avoid too large values for 2^n by scaling the exponent to visualize shapes
n = np.linspace(1, n_max, num=300)

def f_const(n):  # O(1)
    return np.ones_like(n)

def f_log(n):  # O(log n)
    # log base 2, shift to avoid log(0)
    return np.log2(n)

def f_n(n):  # O(n)
    return n

def f_nlogn(n):  # O(n log n)
    return n * np.log2(n)

def f_sq(n):  # O(n^2)
    return n**2

def f_exp(n):  # O(2^n) - scale exponent to keep plot readable
    # We'll compute 2^(n/scale) where scale chosen relative to n_max
    # Choose scale such that at n_max, 2^(n/scale) is not inf; scale = max(1, n_max/20)
    scale = max(1.0, n_max / 20.0)
    return 2 ** (n / scale)

# map labels to functions and LaTeX expressions for display
FUNCTIONS = {
    "O(1)": (f_const, r"f(n) = O(1)"),
    "O(log n)": (f_log, r"f(n) = O(\log n)"),
    "O(n)": (f_n, r"f(n) = O(n)"),
    "O(n log n)": (f_nlogn, r"f(n) = O(n \log n)"),
    "O(n^2)": (f_sq, r"f(n) = O(n^2)"),
    "O(2^n)": (f_exp, r"f(n) = O(2^n)")
}

# -------------------------
# Plotting
# -------------------------
st.subheader("Graphical comparison (interactive)")

fig, ax = plt.subplots(figsize=(10, 5))

color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Plot selection
plotted_any = False
for i, key in enumerate(show_functions):
    func, latex = FUNCTIONS[key]
    y = func(n)
    # For plotting and labeling, avoid NaN/Inf from extremely large values
    y = np.nan_to_num(y, nan=0.0, posinf=np.max(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 1e6)
    label = f"{key} — {latex}"
    ax.plot(n, y, label=key, linewidth=line_width, marker=point_marker if point_marker else None, markevery=max(1, len(n)//25))
    plotted_any = True

if not plotted_any:
    st.info("Select at least one complexity on the left sidebar to see plots.")
else:
    ax.set_xlabel("Input size (n)")
    ax.set_ylabel("Growth Rate f(n)")
    ax.set_title("Growth Comparison of Complexity Functions")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    if use_log_scale:
        # Log scale with formatter to show powers cleanly
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y_val, _: f"{y_val:.0g}"))
    else:
        # autoscale y a bit to avoid overlapping legend
        ax.set_ylim(bottom=0)

    st.pyplot(fig)

    # Allow download of the figure as PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="Download plot as PNG",
        data=buf,
        file_name="asymptotic_plot.png",
        mime="image/png"
    )

# -------------------------
# Mathematical expressions & LaTeX explanation
# -------------------------
st.markdown("---")
st.header("Mathematical notation and Definitions")

st.markdown(r"""
*Asymptotic notations* — brief definitions:

- *Big O (Upper bound):*  
  $f(n) = O(g(n))$ iff $\exists\ c>0,\ n_0\ \text{such that}\ f(n) \le c\cdot g(n)\ \forall n\ge n_0$.  
  (Upper bound; f grows no faster than g up to constant factor.)

- *Big Omega (Lower bound):*  
  $f(n) = \Omega(g(n))$ iff $\exists\ c>0,\ n_0\ \text{such that}\ f(n) \ge c\cdot g(n)\ \forall n\ge n_0$.  
  (Lower bound; f grows no slower than g.)

- *Big Theta (Tight bound):*  
  $f(n) = \Theta(g(n))$ iff $f(n) = O(g(n))$ and $f(n) = \Omega(g(n))$.  
  (Tight bound; f and g grow at the same rate up to constant factors.)
""")

st.markdown("*Examples mapping (three distinct examples)*")
st.markdown(textwrap.dedent("""
1. *O(1)* — Constant time  
   - Example operations: accessing arr[i], reading/writing a variable, returning a value from a function.  
   - Example algorithm: Hash table lookup (average-case) — $T(n)=O(1)$.

2. *O(log n)* — Logarithmic time  
   - Example algorithms: Binary Search on a sorted array, searching in a balanced binary search tree (BST).  
   - Reason: each step cuts the input size multiplicatively.

3. *O(n)* — Linear time  
   - Example algorithms: Single scan (linear search), computing sum of array elements.

4. *O(n log n)* — Linearithmic time  
   - Example algorithms: Merge Sort, Heapsort, QuickSort (average-case).

5. *O(n^2)* — Quadratic time  
   - Example algorithms: Bubble Sort, Insertion Sort (worst-case), selection sort nested loops.

6. *O(2^n)* — Exponential time  
   - Example algorithms: Naive recursive Fibonacci, solving subset-sum by checking all subsets.
"""))

# Show LaTeX expressions nicely in Streamlit
st.markdown("*Display of common function forms (LaTeX):*")
for key in FUNCTIONS:
    _, latex = FUNCTIONS[key]
    st.latex(latex)

# -------------------------
# Hard pseudocode and step-by-step calculation
# -------------------------
st.markdown("---")
st.header("Hard pseudocode example and step-by-step complexity analysis")

st.markdown("""
Consider the following pseudocode (moderately hard): the inner loop depends on i.
""")

pseudo = r"""
function complexProcedure(A[1..n]):
    total = 0
    for i = 1 to n:
        j = 1
        while j <= i:
            // O(1) work
            total = total + A[j]
            j = j * 2
    return total
"""

st.code(pseudo, language="python")

st.markdown("*Reasoning (step-by-step):*")
st.markdown(r"""
We need to count how many constant-time operations happen as n grows.

- For a fixed i, the inner while j <= i loop increments j multiplicatively: j = j * 2.  
  So the inner loop runs about $\lfloor \log_2 i \rfloor + 1$ times (because powers of 2: $1,2,4,8,\dots$ up to $i$).

- The outer loop runs i from 1 to n. So total number of constant operations:

\[
T(n) = \sum_{i=1}^{n} \big( \lfloor \log_2 i \rfloor + 1 \big)
     = \sum_{i=1}^{n} \lfloor \log_2 i \rfloor \;+\; \sum_{i=1}^{n} 1
     = \sum_{i=1}^{n} \lfloor \log_2 i \rfloor \;+\; n
\]

We analyze the sum $\sum_{i=1}^{n} \lfloor \log_2 i \rfloor$.

Group indices by powers of two:
- For $i$ in $[1,1]$, $\lfloor\log_2 i\rfloor = 0$ (1 item)
- For $i$ in $[2,3]$, $\lfloor\log_2 i\rfloor = 1$ (2 items)
- For $i$ in $[4,7]$, $\lfloor\log_2 i\rfloor = 2$ (4 items)
- ...
- For $i$ in $[2^k, 2^{k+1}-1]$, $\lfloor\log_2 i\rfloor = k$ (there are $2^k$ items)

Let $m = \lfloor\log_2 n\rfloor$.
Then:
\[
\sum_{i=1}^{n} \lfloor \log_2 i \rfloor
\;\le\; \sum_{k=0}^{m} k \cdot 2^{k}
\]

We can use the identity (or bound) for the finite sum:
\[
\sum_{k=0}^{m} k 2^{k} = (m-1)2^{m+1} + 2 \quad \text{(closed form; grows like } O(m 2^m))
\]

But note $2^m \le n < 2^{m+1}$, so $m \approx \log_2 n$ and $2^m \le n$.
Thus $\sum_{k=0}^{m} k 2^{k} = O(n \log n)$.

So the dominant term is $O(n \log n)$ from the grouped sum, plus the outer $+ n$ term:

\[
T(n) = O(n \log n) + n = O(n \log n)
\]

Therefore the algorithm runs in *$O(n \log n)$ time*.
""")

st.markdown("*Short summary:* inner loop cost ≈ $\log i$, summing over i=1..n yields $Σ log i = Θ(n log n)`. So the overall complexity is $Θ(n log n)$.")

# -------------------------
# Interactive "show work" mini-calculator (optional)
# -------------------------
st.markdown("---")
st.header("Interactive: numeric preview and quick comparisons")

st.markdown("You can pick a few n values to see concrete numeric values for each complexity (helps intuition).")

# small table generator
sample_ns = st.multiselect(
    "Pick sample n values to evaluate (choose 1..6 values)",
    options=[5, 10, 20, 50, 100, 200, 400, 800],
    default=[10, 50, 100]
)

if sample_ns:
    table = {"n": []}
    for label, (fn, _) in FUNCTIONS.items():
        table[label] = []
    for nv in sample_ns:
        arr_n = np.array([nv], dtype=float)
        table["n"].append(nv)
        for label, (fn, _) in FUNCTIONS.items():
            # evaluate with same scaling rules
            val = fn(arr_n)[0]
            # round for readability
            if val > 1e6:
                table[label].append(f"{val:.3e}")
            else:
                table[label].append(f"{val:.0f}")
    # display as a simple table
    import pandas as pd
    df_preview = pd.DataFrame(table)
    st.table(df_preview.set_index("n"))

# -------------------------
# Final remarks and download
# -------------------------
st.markdown("---")
st.subheader("Project notes & deployment")
st.markdown(textwrap.dedent("""
- This single-file app.py is ready for Streamlit Community Cloud.
- Minimal packages: streamlit, numpy, matplotlib. (Include a requirements.txt with these lines to help deployment.)
- For more realism, the O(2^n) curve above was scaled on the exponent to allow visual comparison across reasonable n ranges; the mathematical notation shown is the standard $2^n$.
"""))

st.markdown("If you'd like, download the *current plot* or save the *numeric preview table* and include this file in your submission ZIP along with requirements.txt and a README.md.")

# end of app