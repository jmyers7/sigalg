import matplotlib.pyplot as plt


def plot_intro_notebook_filtration():
    TWO_CHILD_ATOMS = {"02", "12", "22"}
    level0 = [""]
    level1 = ["0", "1", "2"]
    level2 = [a + b for a in "012" for b in "012"]
    level3 = []
    children_of = {atom: [] for atom in level0 + level1 + level2}

    for a in level1:
        children_of[""].append(a)

    for a in level1:
        for b in "012":
            children_of[a].append(a + b)

    for a in level2:
        digits = "01" if a in TWO_CHILD_ATOMS else "012"
        for d in digits:
            child = a + d
            children_of[a].append(child)
            level3.append(child)

    all_nodes = level0 + level1 + level2 + level3

    pos = {}
    _leaf_counter = [0]

    def layout(atom):  # noqa: D103
        kids = children_of.get(atom, [])
        depth = len(atom)
        if not kids:
            y = -_leaf_counter[0]
            _leaf_counter[0] += 1
            pos[atom] = (depth, y)
            return y
        ys = [layout(child) for child in kids]
        y = sum(ys) / len(ys)
        pos[atom] = (depth, y)
        return y

    layout("")

    def label(atom):  # noqa: D103
        if atom == "":
            return r"$\Omega$"
        return rf"$A_{{{atom}}}$"

    fig, ax = plt.subplots(figsize=(11, 9))

    for parent, kids in children_of.items():
        x0, y0 = pos[parent]
        for child in kids:
            x1, y1 = pos[child]
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=0.8, zorder=1)

    for atom in all_nodes:
        x, y = pos[atom]
        ax.text(
            x,
            y,
            label(atom),
            ha="center",
            va="center",
            fontsize=14,
            color="black",
            zorder=2,
            bbox={
                "boxstyle": "round,pad=0.3,rounding_size=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.8,
            },
        )

    ax.set_xlim(-0.5, 3.5)
    ax.axis("off")
    ax.set_title(
        r"MacKay's optimal weighing strategy, translated to a filtration of $\sigma$-algebras",
        fontsize=14,
        pad=20,
    )
    fig.tight_layout()
    plt.show()
