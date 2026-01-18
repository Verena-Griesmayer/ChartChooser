from dataclasses import dataclass
from pathlib import Path
import csv, re
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates


# picks one random color
def random_color() -> str:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return random.choice(palette)

# picks n random colors
def random_colors(n: int) -> List[str]:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [random.choice(palette) for _ in range(n)]


# detects the seperator with the first 10 lines
def get_seperator(sample: str) -> str:
    candidates = [";", ",", "\t", "|"]
    counts = {d: sample.count(d) for d in candidates}
    return max(counts, key=counts.get)


# reads the data of the file and returns headers and data
def read_data(path: Path, encoding: str = "utf-8") -> Tuple[str, List[str], List[List[str]]]:
    text = path.read_text(encoding=encoding, errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError("File must contain at least a title line and a header line.")

    title = lines[0].strip()

    delimiter = get_seperator("\n".join(lines[1:11])) # gets first 10 lines to detect seperator
    reader = csv.reader(lines[1:], delimiter=delimiter)

    rows = list(reader)
    header = [h.strip() for h in rows[0]]
    data = [[cell.strip() for cell in row] for row in rows[1:] if row]

    return title, header, data


# DETECT NUMBERS
_num_re = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*$") # Regex for detecting numbers


def parse_number(s: str) -> Optional[float]:
    if s is None:
        return None

    s = s.strip() # Remove white spaces at the end and begin

    if s == "":
        return None

    s = s.replace(" ", "") # Remove white spaces in the middle

    if not _num_re.match(s): # if not matching with Regex
        return None

    if "," in s and "." not in s: # either . or , not both
        s = s.replace(",", ".")
    else:
        pass

    try:
        return float(s)
    except ValueError:
        return None


# DETECT DATE
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
]


def parse_date(s: str) -> Optional[datetime]:
    s = s.strip() # Remove whitespaces at end and begin
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# Detect either categorical, numeric or datetime
def infer_column_type(values: List[str]) -> str:
    non_empty = [v for v in values if v.strip() != ""] # Removes unnecessary data
    if not non_empty:
        return "categorical"

    num_hits = 0
    date_hits = 0

    for v in non_empty:
        if parse_number(v) is not None:
            num_hits += 1
        elif parse_date(v) is not None:
            date_hits += 1

    n = len(non_empty)

    if num_hits / n >= 0.8:
        return "numeric"
    if date_hits / n >= 0.8:
        return "datetime"
    return "categorical"


def infer_schema(header: List[str], rows: List[List[str]]) -> Dict[str, str]:
    cols = list(zip(*rows)) if rows else [] # Separates Columns and puts them together in a list
    schema = {}
    for i, name in enumerate(header):
        values = list(cols[i]) if i < len(cols) else []
        schema[name] = infer_column_type(values) # Analyse data to get type
    return schema


# Generates DatasetProfile object from header and rows
@dataclass
class DatasetProfile:
    header: List[str]
    schema: Dict[str, str]
    n_rows: int
    n_cols: int


def profile_dataset(header: List[str], rows: List[List[str]]) -> DatasetProfile:
    schema = infer_schema(header, rows)
    return DatasetProfile(header=header, schema=schema, n_rows=len(rows), n_cols=len(header))


# Class for the real analysis
@dataclass
class ChartDecision:
    chart_type: str                 # bar|line|scatter|hist|table
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    params: Dict[str, Any] = None


# Decision for the smartest diagramm
def choose_chart(p: DatasetProfile) -> ChartDecision:
    types = list(p.schema.values())
    names = p.header

    numeric_cols = [c for c in names if p.schema[c] == "numeric"]
    cat_cols = [c for c in names if p.schema[c] == "categorical"]
    date_cols = [c for c in names if p.schema[c] == "datetime"]

    if len(numeric_cols) == 1 and (len(cat_cols) == 0 and len(date_cols) == 0):
        return ChartDecision("hist", x_col=numeric_cols[0], params={"bins": 20})

    if len(date_cols) >= 1 and len(numeric_cols) >= 1:
        return ChartDecision("line", x_col=date_cols[0], y_col=numeric_cols[0], params={})

    if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        decision = ChartDecision("bar", x_col=cat_cols[0], y_col=numeric_cols[0], params={})

        decision.params["top_n"] = 15
        decision.params["sort"] = "desc"
        return decision

    if len(numeric_cols) >= 2:
        return ChartDecision("scatter", x_col=numeric_cols[0], y_col=numeric_cols[1], params={})

    return ChartDecision("table", params={})

# Plotting
def column_index(header: List[str], name: str) -> int:
    try:
        return header.index(name)
    except ValueError:
        raise ValueError(f"Column '{name}' not found. Available: {header}")

def extract_column(rows: List[List[str]], idx: int) -> List[str]:
    out = []
    for r in rows:
        if idx < len(r):
            out.append(r[idx])
        else:
            out.append("")
    return out

def plot_decision(header: List[str], rows: List[List[str]], decision: ChartDecision, title: Optional[str] = None) -> None:
    params = decision.params or {}

    if decision.chart_type == "table":
        print("No clear diagram detected. Show preview as a table (first 10 rows).")
        print("; ".join(header))
        for r in rows[:10]:
            print("; ".join(r))
        return

    if decision.chart_type == "hist":
        x_idx = column_index(header, decision.x_col)
        raw = extract_column(rows, x_idx)
        nums = [parse_number(v) for v in raw]
        nums = [v for v in nums if v is not None]

        if not nums:
            raise ValueError("No numerical values found for histogram.")

        plt.figure()
        plt.hist(nums, bins=params.get("bins", 20), color=random_color())
        plt.xlabel(decision.x_col)
        plt.ylabel("Frequency")
        plt.title(title or f"Histogramm: {decision.x_col}")
        plt.tight_layout()
        plt.show()
        return

    if decision.chart_type == "scatter":
        x_idx = column_index(header, decision.x_col)
        y_idx = column_index(header, decision.y_col)
        x_raw = extract_column(rows, x_idx)
        y_raw = extract_column(rows, y_idx)

        xs, ys = [], []
        for a, b in zip(x_raw, y_raw):
            x = parse_number(a)
            y = parse_number(b)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)

        if not xs:
            raise ValueError("No matching numerical pairs found for Scatter.")

        plt.figure()
        plt.scatter(xs, ys, color=random_color())
        plt.xlabel(decision.x_col)
        plt.ylabel(decision.y_col)
        plt.title(title or f"Scatter: {decision.y_col} vs {decision.x_col}")
        plt.tight_layout()
        plt.show()
        return

    if decision.chart_type == "line":
        x_idx = column_index(header, decision.x_col)
        y_idx = column_index(header, decision.y_col)
        x_raw = extract_column(rows, x_idx)
        y_raw = extract_column(rows, y_idx)

        points = []
        for a, b in zip(x_raw, y_raw):
            dt = parse_date(a)
            y = parse_number(b)
            if dt is None or y is None:
                continue
            points.append((dt, y))

        if not points:
            raise ValueError("No matching date/number pairs found for line chart.")

        points.sort(key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        plt.figure()
        plt.plot(xs, ys, color=random_color())
        plt.xlabel(decision.x_col)
        plt.ylabel(decision.y_col)
        plt.title(title or f"Zeitreihe: {decision.y_col}")

        ax = plt.gca()

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m."))  # z.B. 01.01.

        plt.gcf().autofmt_xdate(rotation=45, ha="right")  # lesbar drehen
        plt.tight_layout()
        plt.show()
        return

    if decision.chart_type == "bar":
        x_idx = column_index(header, decision.x_col)
        y_idx = column_index(header, decision.y_col)
        x_raw = extract_column(rows, x_idx)
        y_raw = extract_column(rows, y_idx)

        pairs = []
        for a, b in zip(x_raw, y_raw):
            y = parse_number(b)
            if a.strip() == "" or y is None:
                continue
            pairs.append((a.strip(), y))

        if not pairs:
            raise ValueError("No matching category/number pairs found for bar chart.")

        # sort
        if params.get("sort") == "desc":
            pairs.sort(key=lambda t: t[1], reverse=True)
        elif params.get("sort") == "asc":
            pairs.sort(key=lambda t: t[1])

        # top_n
        top_n = params.get("top_n")
        if isinstance(top_n, int) and top_n > 0 and len(pairs) > top_n:
            pairs = pairs[:top_n]

        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]

        plt.figure()
        plt.bar(labels, values, color=random_colors(len(values)))
        plt.xlabel(decision.x_col)
        plt.ylabel(decision.y_col)
        plt.title(title or f"{decision.y_col} zu {decision.x_col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return

    raise ValueError(f"Unknown chart_type: {decision.chart_type}")


# All-in-one function
def analyze_and_plot(path: Path) -> None:
    title, header, rows = read_data(path)
    prof = profile_dataset(header, rows)
    decision = choose_chart(prof)

    print("Detected scheme:", prof.schema)
    print("Diagram decision:", decision)
    print("Chart title:", title)

    plot_decision(header, rows, decision, title=title)


def main():
    data_file1 = Path(__file__).parent.parent / "Test_Data" / "data01.csv"
    data_file2 = Path(__file__).parent.parent / "Test_Data" / "data02.csv"
    data_file3 = Path(__file__).parent.parent / "Test_Data" / "data03.csv"
    data_file4 = Path(__file__).parent.parent / "Test_Data" / "data04.csv"
    data_file5 = Path(__file__).parent.parent / "Test_Data" / "data05.csv"
    analyze_and_plot(data_file1)
    analyze_and_plot(data_file2)
    analyze_and_plot(data_file3)
    analyze_and_plot(data_file4)
    analyze_and_plot(data_file5)


if __name__ == "__main__":
    main()


