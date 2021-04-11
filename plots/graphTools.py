# coding: 8859
import argparse
import pandas as pds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import textwrap
from matplotlib.backends.backend_pdf import PdfPages


def openfile(path="./plots/data/perf_data.csv", sepa=";"):
    try:
        return pds.read_csv(path, sep=sepa)
    except FileNotFoundError:
        print("File not found: ", path, file=sys.stderr)
        sys.exit(1)

# Donne tous les champs de df qui ne sont pas listés


def complementaryCols(listeAttr, df):
    return [i for i in list(df.columns) if i not in listeAttr]


def constantCols(df):
    return df.columns[df.eq(df.iloc[0]).all()].tolist()


def allData(name, df):  # Donne toutes les valeurs possibles de la colonne "name" dans "df"
    return df[name].unique()


def shuffle(u, v, sepa=" "):
    s = ""
    for i in zip(u, v):
        s += str(i[0]) + "=" + str(i[1]) + sepa
    return s

# Cree les nom avec toutes les valeurs non utilisees pour le reste du graphique


def creationLegende(datasForGrapheNames, df):

    attr = complementaryCols(['time', 'ref'] + datasForGrapheNames +
                             [i for i in list(df.columns) if df[i].nunique() == 1], df)

    if attr == []:
        attr = ['kernel']

    df['legend'] = df[attr].apply(
        lambda row: shuffle(attr, row.values.tolist()), axis=1)

# Donne les valeurs constantes de "df" a travers un dictionnaire


def texteParametresConstants(df, texte=""):
    const = constantCols(df)
    string = ""
    # Donne les valeurs des constantes dans l'ordre des colonnes
    dataConst = [df[i].iloc[0] for i in const]
    for i in range(len(const)):
        if dataConst[i] != "none" and str(dataConst[i]) != "nan":
            string = string + str(const[i]) + "=" + str(dataConst[i]) + " "
    return texte + " " + string


def nombreParametresConstants(df):
    return len(constantCols(df))


def creationRefSpeedUp(df, args):  # Automatise la creation du speedup

    group = ['machine', 'size', 'kernel', 'iterations', 'arg']
    group = [attr for attr in group if attr not in args.delete]

    if args.RefTimeVariants == "":
        refDF = df[df.threads == 1].reset_index(drop=True)
    else:
        refDF = df[df.variant.isin(args.RefTimeVariants)
                   & df.threads == 1].reset_index(drop=True)
    if refDF.empty:
        print("No reference to compute speedUP")
        exit()

    for i in complementaryCols(group + ['time'], refDF):
        del refDF[i]

    refDF = refDF.loc[refDF.groupby(
        group).time.idxmin()].reset_index(drop=True)
    refDF = refDF.rename(columns={'time': 'refTime'})

    df = df.merge(refDF, how='inner')

    df['speedup'] = df['refTime'] / df['time']

    if args.noRefTime:
        del df['refTime']

    return df


def creerGraphique(df, args):
    constNum = nombreParametresConstants(df)
    datasForGrapheNames = [args.x, args.y, args.col, args.row]
    creationLegende(datasForGrapheNames, df)

    if args.y == "time":
        df['time'] = df['time'] / 1000
        df.rename(columns={'time': 'time (ms)'}, inplace=True)
        args.y = "time (ms)"

    if not args.no_sort:
        df = df.sort_values(by=args.y, ascending=False)

    if (args.plottype == 'lineplot'):
        g = sns.FacetGrid(df, row=args.row, col=args.col, hue="legend", sharex='col', sharey='row',
                          height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect)

        g.map(sns.lineplot, args.x, args.y, err_style="bars", marker="o")
        g.set(xscale=args.xscale)
        g.add_legend()
        if args.x == 'threads':
            g.set(xlim=(0, None))
    else:
        g = sns.catplot(data=df, x=args.x, y=args.y, row=args.row, col=args.col, hue="legend",
                        kind=args.kind, sharex='col', sharey='row',
                        height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect)

    if args.font_scale != 1.0:
        sns.set(font_scale=args.font_scale)
    g.set(yscale=args.yscale)

    if constNum == 0:
        titre = (u'Courbe de {y} en fonction de {x}').format(
            x=args.x, y=args.y)
    else:
        titre = (u'{cons}').format(x=args.x, y=args.y,
                                   cons=texteParametresConstants(df, ""))
    if not args.hideParameters:
        plt.subplots_adjust(top=args.adjustTop)
        g.fig.suptitle(titre, wrap=True)
    else:
        print(titre)
    return g


def parserArguments(argv):
    global parser, args
    parser = argparse.ArgumentParser(
        argv,
        description='''Process performance plots.
        The labels of the options are similar to those of
        easypap for all that relates to data selection,
        and to those of seaborn for all aspects of graphic
        layout https://seaborn.pydata.org/introduction.html''')

    all = ["size", "iterations", "kernel", "variant", "threads",
           "nb_tiles", "schedule", "label", "machine", "tile_size", "tileh", "tilew", "arg"]

    parser.add_argument("-x", choices=all+["custom"], default="threads")
    parser.add_argument(
        "-y", choices=["time", "speedup", "throughput", "custom"], default="speedup")

    parser.add_argument('-rtv', '--RefTimeVariants',
                        action='store', nargs='+',
                        help="list of variants to take into account to compute the speedUP RefTimes",
                        default="")

    parser.add_argument("-C", "--col", choices=all+["custom"], default=None)
    parser.add_argument("-R", "--row", choices=all+["custom"], default=None)

    parser.add_argument('-of', '--output',
                        action='store', nargs='?',
                        help='Filename to output the plot',
                        const='plot.pdf',
                        default='plot.pdf')

    parser.add_argument('-if', '--input',
                        action='store', nargs='?',
                        help="Data's filename",
                        const=os.getcwd() + "/plots/data/perf_data.csv",
                        default=os.getcwd() + "/plots/data/perf_data.csv")

    parser.add_argument('-k', '--kernel',
                        action='store', nargs='+',
                        help="list of kernels to plot",
                        default="")

    parser.add_argument('-t', '--threads',
                        action='store', nargs='+',
                        help="list of numbers of threads to plot",
                        default="")

    parser.add_argument('--delete',
                        action='store', nargs='+',
                        help="delete a column before proceeding data",
                        choices=all,
                        default=""
                        )

    parser.add_argument('-v', '--variant',
                        action='store', nargs='+',
                        help="list of variants to plot",
                        default="")

    parser.add_argument('-th', '--tileh',
                        action='store', nargs='+',
                        help="list of tile heights to plot",
                        default="")

    parser.add_argument('-tw', '--tilew',
                        action='store', nargs='+',
                        help="list of tile widths to plot",
                        default="")

    parser.add_argument('-nt', '--nb_tiles',
                        action='store', nargs='+',
                        help="list of nb_tiles to plot",
                        default="")


#    parser.add_argument('-ts', '--tile',
#                        action='store', nargs='*',
#                        help="print tile sizes rather than nb_tiles / list of tiles to plot",
#                        default=None)

    parser.add_argument('-m', '--machine',
                        action='store', nargs='+',
                        help="list of machines to plot",
                        default="")

    parser.add_argument('-i', '--iterations',
                        action='store', nargs='+',
                        help="list of iterations to plot",
                        default="")

    parser.add_argument('-sc', '--schedule',
                        action='store', nargs='+',
                        help="list of schedule policies to plot",
                        default="")

    parser.add_argument('-s', '--size',
                        action='store', nargs='+',
                        help="list of sizes to plot",
                        default="")

    parser.add_argument('-lb', '--label',
                        action='store', nargs='+',
                        help="list of labels to plot",
                        default="")

    parser.add_argument('--height',
                        action='store',
                        type=int,
                        help="to set the height of each subgraph",
                        default=4)

    parser.add_argument('--hideParameters',
                        action='store_true',
                        help="to hide constant parameters",
                        default=False)

    parser.add_argument('--legendInside',
                        action='store_true',
                        help="to print the legend inside the graph",
                        default=False)

    parser.add_argument("--no_sort",
                        action='store_true',
                        help="sort data following y",
                        default=False)

    parser.add_argument('--adjustTop',
                        action='store',
                        type=float,
                        help="to adjust the space for the suptitle",
                        default=.9)

    parser.add_argument('--aspect',
                        action='store',
                        type=float,
                        help="to adjust the ratio length/height",
                        default=1.1)

    parser.add_argument('--font_scale',
                        action='store',
                        type=float,
                        help="to adjust the font of the title and the legend",
                        default=1.0)

    parser.add_argument('--noRefTime',
                        action='store_true',
                        help="do not print reftime in legend",
                        default=False)

    parser.add_argument('--yscale',
                        choices=["linear", "log", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--xscale',
                        choices=["linear", "log", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--plottype',
                        choices=['lineplot', 'catplot'],
                        action='store',
                        default="lineplot")

    parser.add_argument('--kind',
                        choices=["strip", "swarm", "box", "violin",
                                 "boxen", "point", "bar", "count"],
                        help="kind of barplot (see sns catplot)",
                        action='store',
                        default="swarm")

    args = parser.parse_args()
    return args


def lireDataFrame(args):
    # Lecture du fichier d'experiences:
    df = openfile(args.input, sepa=";")

    if args.kernel != "":
        df = df[df.kernel.isin(args.kernel)].reset_index(drop=True)

    if args.iterations != "":
        df = df[df.iterations.isin(args.iterations)].reset_index(drop=True)

    if args.size != "":
        df = df[df["size"].isin(args.size)].reset_index(drop=True)

    if args.machine != "":
        df = df[df.machine.isin(args.machine)].reset_index(drop=True)

    if args.delete != []:
        for attr in args.delete:
            del df[attr]

    if args.y == "speedup":
        df = creationRefSpeedUp(df, args)

    if args.label != "":
        df = df[df.label.isin(args.label)].reset_index(drop=True)

    if args.schedule != "":
        df = df[df.schedule.isin(args.schedule)].reset_index(drop=True)

    if args.threads != "":
        df = df[df.threads.isin(args.threads)].reset_index(drop=True)

    if args.variant != "":
        df = df[df.variant.isin(args.variant)].reset_index(drop=True)

    if args.tileh != "":
        df = df[df.tileh.isin(args.tileh)].reset_index(drop=True)

    if args.tilew != "":
        df = df[df.tilew.isin(args.tilew)].reset_index(drop=True)

    if args.nb_tiles == "":
        if not ('tileh' in [args.col, args.row, args.x] or 'tilew' in [args.col, args.row, args.x]):
            df['tile'] = df.tileh.map(str) + '$\\times$' + df.tilew.map(str)
            del df['tileh']
            del df['tilew']
    else:
        df['nb_tileh'] = df['size'] // df['tileh']
        df['nb_tilew'] = df['size'] // df['tilew']
        df = df[df.nb_tilew.isin(args.nb_tiles)].reset_index(drop=True)
        # df = df[df.nb_tileh.isin(args.nb_tiles)].reset_index(drop=True)
        df = df[df.nb_tileh == df.nb_tilew].reset_index(drop=True)
        df['nb_tiles'] = df['nb_tileh']
        del df['tileh']
        del df['tilew']
        del df['nb_tileh']
        del df['nb_tilew']

    if args.y == "throughput":
        args.y = 'throughput (MPixel / s)'
        df[args.y] = (df['size'] ** 2) * df['iterations'] / df['time']

    if df.empty:
        print("No data")
        exit()

    # remove empty columns
    return df.dropna(axis=1, how='all')


def engeristrerGraphique(fig):
    pp = PdfPages(args.output)
    plt.savefig(pp, format='pdf')
    pp.close()
