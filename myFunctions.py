def myDateDistri(s, nBars=100, ax=None):
    """
    plot empirical distribution of a datetime column

    parameters :
    ------------
    s - Series : datetime dtype
    n_bars - int : maximum number of bars desired. By default 15
    ax - axes : By default None
    """
    # imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # put s in a dataframe
    df = s.copy().to_frame()

    # extract the date and set a datetimeIndex with it, creating a timeSeries
    df["date"] = pd.to_datetime(s.dt.date)
    df.set_index("date", inplace=True)

    # resample the timeSeries

    # investigate on the date period to choose frequency of the resample, and xtick labels
    period = (s.max() - s.min()).ceil("d").days
    if period > 365:
        # adapt period
        periodMonths = period / 365 * 12
        # choose frequence
        digitFreq = int(np.ceil(periodMonths / nBars))
        freq = str(digitFreq) + "M"
        # set xticklabels date_format
        date_format = "%b-%Y"

    else:
        digitFreq = int(np.ceil(period / nBars))
        # choose frequence
        freq = str(digitFreq) + "d"
        # set xticklabels date_format
        date_format = "%d-%b"

    # resample
    df = df.resample(freq).count()

    # plot
    sns.barplot(x=df.index, y=df[s.name], ax=ax)

    # adjust x tick labels
    labels = [e.strftime(date_format) for e in df.index]
    if not ax:
        ax = plt.gca()
    ax.set_xticklabels(labels, rotation=45, ha="right")


def myDescribe(dataframe):
    """displays a Pandas .describe with options : quantitaves columns, qualitatives columns, all columns.
    If a dict is given as an input : {"df1Name" : df1, "df2Name" : df2, etc.}, one can choose the dataframe

    parameters :
    ------------
    dataframe : Pandas dataframe or a Dict

    """

    import ipywidgets as widgets  # import library
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # def main function
    def myDescribeForOnlyOneDf(df):
        # if df is a dictionnary key, we get its value
        if type(df) == str:
            df = dataframe[df]

        # widget with .describe() display options
        widDescribe = widgets.RadioButtons(
            options=["quantitative", "qualitative", "all"],
            value="all",
            description="Which features :",
            disabled=False,
            style={"description_width": "initial"},
        )

        # widget to select column
        widColList = widgets.Dropdown(
            options={"all": None} | {col: col for col in list(df.columns)},
            value=None,
            description="Which column :",
            disabled=False,
            style={"description_width": "initial"},
        )

        def handle_widDescribe_change(
            change,
        ):  # linking col dropdwn with the type of describe display option
            if change.new == "qualitative":
                widColList.options = {"all": None} | {
                    col: col
                    for col in list(df.select_dtypes(["O", "category"]).columns)
                }
            if change.new == "quantitative":
                widColList.options = {"all": None} | {
                    col: col
                    for col in list(
                        df.select_dtypes(
                            [
                                "float64",
                                "float32",
                                "float16",
                                "int64",
                                "int32",
                                "int16",
                                "int8",
                                "uint8",
                            ]
                        ).columns
                    )
                }
            if change.new == "all":
                widColList.options = {"all": None} | {
                    col: col for col in list(df.columns)
                }

        widDescribe.observe(handle_widDescribe_change, "value")

        # sub function used in final output
        def describeFunct(df, whichTypes, columnName=None):
            if whichTypes == "qualitative":
                include = ["O", "category"]
                exclude = [
                    "float64",
                    "float32",
                    "float16",
                    "int64",
                    "int32",
                    "int16",
                    "int8",
                    "uint8",
                ]
            elif whichTypes == "quantitative":
                include = None
                exclude = None
            elif whichTypes == "all":
                include = "all"
                exclude = None
            if columnName:
                df = df[[columnName]]
            describeTable = df.describe(include=include, exclude=exclude)
            # add dtypes
            describeTable.loc["dtype"] = describeTable.apply(
                lambda s: df[s.name].dtype
            ).values.tolist()
            describeTable.loc["%NaN"] = describeTable.apply(
                lambda s: (round(df[s.name].isna().mean() * 100, 1)).astype(str) + "%"
            ).values.tolist()
            describeTable = pd.concat(
                [describeTable.iloc[-1:], describeTable.iloc[:-1]]
            )

            # decide which kind of display

            # for columns other than "O", we can plot distribution next to .describe() table
            if columnName and df[columnName].dtype.kind not in "O":
                # create fig and 2 axes, one for the table, one for the plot
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, width_ratios=[1, 4], figsize=(14, 4)
                )
                # set lines colors, "grey" every other line
                colors = [
                    "#F5F5F5" if i % 2 == 1 else "w" for i in range(len(describeTable))
                ]
                # plot table
                ax1.table(
                    cellText=describeTable.values,
                    rowLabels=describeTable.index,
                    bbox=[0, 0, 1, 1],
                    colLabels=describeTable.columns,
                    cellColours=[[color] for color in colors],
                    rowColours=colors,
                )
                ax1.axis(False)
                # plot a box plot if column is numerical and not datetime
                if df[columnName].dtype.kind not in "mM":
                    sns.boxplot(data=df, x=columnName, ax=ax2)
                # if datatime, use myDateDistri function
                else:
                    myDateDistri(s=df[columnName], ax=ax2)
                plt.show()

            else:
                display(describeTable)

        # output
        out = widgets.interactive_output(
            describeFunct,
            {
                "df": widgets.fixed(df),
                "whichTypes": widDescribe,
                "columnName": widColList,
            },
        )
        display(widgets.HBox([widDescribe, widColList]), out)

    # if input is a dataframe, use above function
    if type(dataframe) != dict:
        myDescribeForOnlyOneDf(dataframe)

    # if input is a dict, add a widget to select a dataframe
    else:
        widDfList = widgets.Dropdown(
            options=list(dataframe.keys()),
            value=list(dataframe.keys())[0],
            description="Which dataframe :",
            disabled=False,
            style={"description_width": "initial"},
        )

        out = widgets.interactive_output(myDescribeForOnlyOneDf, {"df": widDfList})
        display(widDfList, out)


def myHead(dataframe):
    """displays a Pandas .head with an option to select "n"
    If a dict is given as an input : {"df1Name" : df1, "df2Name" : df2, etc.}, one can choose the dataframe

    parameters :
    ------------
    dataframe : Pandas dataframe or a Dict

    """
    # import libraries
    import ipywidgets as widgets
    import pandas as pd

    # def function to display .head() with a widget to select "n"
    def myHeadForOneDf(df):
        # if df is a dictionnary key, we get its value
        if type(df) == str:
            df = dataframe[df]

        # widget for selecting the "n" parameter
        widN = widgets.BoundedIntText(
            value=15,
            min=1,
            max=60,
            description="Number of lines : ",
            disabled=False,
            style={"description_width": "initial"},
        )

        # def internal funct
        def myHead(df, n):
            display(df.head(n))

        # use interactive output
        out = widgets.interactive_output(myHead, {"df": widgets.fixed(df), "n": widN})
        display(widN, out)

    # if input is a dataframe, use above function
    if type(dataframe) != dict:
        myHeadForOneDf(dataframe)

    # if input is a dict, add a widget to select the dataframe
    else:
        widDfList = widgets.Dropdown(
            options=list(dataframe.keys()),
            value=list(dataframe.keys())[0],
            description="Which dataframe :",
            disabled=False,
            style={"description_width": "initial"},
        )

        out = widgets.interactive_output(myHeadForOneDf, {"df": widDfList})
        display(widDfList, out)


def distribRidgePlot(
    df,
    categFeatureName,
    numFeatureName,
    palette=None,
    overlap=0.5,
    order=None,
    zoomInterval=None,
    clip=None,
):
    """
    draw a ridgplot, empirical distributions of a numerical column, one for each value of a categorical column

    parameters
    ----------
    df - dataframe
    categFeatureName - string
    numFeatureName - srting

    optionnal
    ---------
    palette : dictionnary, with features names for keys and colors for values. By default None
    overlap : float, to adjust space between facets/axes. Default = 0.5
    order : list of labels of the categorical feature, in a given order
    zoomInterval : tuple or list, zoom interval on x axis. By default None
    clip : tuple or list, kdeplot interval


    """
    # imports
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # set theme
    sns.set_theme(
        style="white",  # seaborn white style
        rc={
            "axes.facecolor": (
                0,
                0,
                0,
                0,
            ),  # set facecolor with alpha = 0, so the background is transparent
            "axes.linewidth": 2,
        },  # thick line width
    )

    # config the order of categories on the grid
    if order:
        myRowOrder = order
    else:
        myRowOrder = (
            df.groupby(categFeatureName, observed=True)[numFeatureName]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )

    # config palette
    nbLabels = df[categFeatureName].nunique()
    if palette:
        if type(palette) == dict:
            myPalette = palette
        elif type(palette) == str:
            myPalette = {
                lab: c
                for lab, c in zip(
                    myRowOrder, sns.color_palette(palette, n_colors=nbLabels)
                )
            }
        else:
            myPalette = {
                lab: c
                for lab, c in zip(myRowOrder, sns.color_palette(n_colors=nbLabels))
            }
    else:
        myPalette = {
            lab: c for lab, c in zip(myRowOrder, sns.color_palette(n_colors=nbLabels))
        }

    # initiate a FacetGrid object
    g = sns.FacetGrid(
        data=df,
        row=categFeatureName,  # create one subplot for each category of feature
        hue=categFeatureName,  # use a different color plot for each subplot
        palette=myPalette,  # use given palette
        row_order=myRowOrder,  # order the categories/levels
        height=1.2,  # height of each facet
        aspect=9,  # height ratio to give the width of each facet
    )

    # draw density plots
    # ones filled with a color
    g.map_dataframe(  # apply plotting function to each facet
        sns.kdeplot,  # density function
        x=numFeatureName,  # use with our numerical feature
        fill=True,  # fill area under plot
        alpha=1,
        clip=clip,
    )
    # ones with only a black line
    g.map_dataframe(sns.kdeplot, x=numFeatureName, color="black", clip=clip)

    # set main title
    plt.suptitle(
        "Distibution of  '"
        + numFeatureName
        + "' \n for each '"
        + categFeatureName
        + "'"
    )

    # set facets titles
    # plots will overlap so we need to place titles in an unusual place
    def writeCategory(x, color, label):  # create a plotting function to write titles
        ax = plt.gca()  # get current axes
        ax.text(
            x=0,  # left of the distribution plot
            y=0.05 / overlap,  # above x axis line
            s=label,  # write current category/label of categFeatureName of the facet
            color=color,  # use the current color of the facet
            fontsize=12,  # def texte size
            ha="left",
            va="center",  # vertical and horizontal alignments
            transform=ax.transAxes,  # use coordinate system of the Axes,
            fontweight="bold",
        )

    g.map(writeCategory, numFeatureName)  # draw our titles on each facet
    g.set_titles("")  # get rid of classic facet titles

    # make plots overlap
    g.fig.subplots_adjust(hspace=-overlap)

    # set ticks and ticklabels
    g.set(
        ylabel="",  # we do not need the y label (always the same)
        yticks=[],  # focus on shape, no need for y ticks
        xlabel=numFeatureName,  # set x label
        xlim=zoomInterval,  # zooming if asked
    )
    g.despine(left=True)
    # get rid of y axis


def myPCA(df, q, ACPfeatures=None):
    """
    run scaling preprocessing, using sklearn.preprocessing.StandardScaler,
    and PCA, using scikit learn sklearn.decomposition.PCA

    parameters :
    ------------
    df : DataFrame on which we want to run the PCA
    q : number of components of the PCA

    optionnal parameters :
    ----------------------
    ACPfeatures = list of columns names of df used for PCA. By default None (in that case : all dtype 'float64' columns names)

    outputs :
    ---------
    X_scaled : values of scaled df
    Xidx : index of df
    Xfeatures : columns names
    dfPCA : PCA fitted with df


    """
    # imports
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # stores values and index
    Xfeatures = (
        ACPfeatures
        if ACPfeatures
        else [col for col in df.columns.tolist() if df[col].dtype.kind in "biufc"]
    )
    PCAdf = df.copy()[Xfeatures]
    X = PCAdf.values
    Xidx = PCAdf.index

    # scale
    scaler = StandardScaler()  # instantiate
    X_scaled = scaler.fit_transform(X)  # fit transform X

    # PCA
    dfPCA = PCA(n_components=q)
    dfPCA.fit(X_scaled)

    return X_scaled, Xidx, Xfeatures, dfPCA


def PCA_scree_plot(pca):
    """
    draw the eigen values scree plot of a given fitted pca

    parameter : fitted sklearn.decomposition.PCA
    """
    import pandas as pd
    import numpy as np

    # initiate dataframe
    screeDf = pd.DataFrame(index=["F" + str(k + 1) for k in range(pca.n_components)])
    # explained variance ratio in percentage
    screeDf["Variance expliquée"] = (pca.explained_variance_ratio_).round(2)
    # cumsum
    screeDf["Variance expliquée cumulée"] = (
        screeDf["Variance expliquée"].cumsum().round(2)
    )

    # plot
    import seaborn as sns

    sns.set_theme()

    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.bar(screeDf, y="Variance expliquée", text_auto=True)
    fig2 = go.Scatter(
        y=screeDf["Variance expliquée cumulée"],
        x=screeDf.index,
        mode="lines+markers",
        showlegend=False,
        name="",
    )
    fig.add_trace(fig2)

    fig.layout.yaxis.tickformat = ",.0%"

    for idx, val in screeDf["Variance expliquée cumulée"].iloc[1:].items():
        fig.add_annotation(
            y=val,
            x=idx,
            text=str(round(val * 100)) + "%",
            showarrow=False,
            yshift=10,
            xshift=-10,
        )

    fig.update_layout(
        height=800,
        title_text="Eboulis des valeurs propres",
        xaxis_title="Composante principale",
        yaxis_title="Valeurs propres - variance expliquée",
    )

    fig.show()


def pcaCorrelationMatrix(pca, PcafeaturesNames, additionnalVariable=None):
    """
    return the correlation matrix 'features <-> loadings', dataframe
    Parameters :
    -----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    PcafeaturesNames : list or tuple : list of pca features names

    Optional parameters :
    ---------------------
    additionnalVariable : list or tuple containing elements to add another variable to the matrix (i.e. another row)
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series object
    """
    import pandas as pd
    import numpy as np

    matrix = pca.components_.T * np.sqrt(pca.explained_variance_)
    dfMatrix = pd.DataFrame(
        matrix,
        index=PcafeaturesNames,
        columns=[
            "F"
            + str(i + 1)
            + " ("
            + str(round(100 * pca.explained_variance_ratio_[i], 1))
            + "%)"
            for i in range(pca.n_components_)
        ],
    )

    if additionnalVariable:
        X_scaled = additionnalVariable[0]
        addVarSeries = additionnalVariable[1]

        C = pd.DataFrame(
            pca.transform(X_scaled), index=addVarSeries.index, columns=dfMatrix.columns
        )
        corrAddVar = C.corrwith(addVarSeries, axis=0)

        dfMatrix.loc["Add Var (" + addVarSeries.name + ")"] = corrAddVar
    return dfMatrix


def heatPcaCorrelationMatrix(
    pca, PcafeaturesNames, additionnalVariable=None, figsize=(10, 5)
):
    """
    display a PCA correlation matrix in a Seaborn Heatmap way

    parameters :
    ----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    PcafeaturesNames : list or tuple : list of pca features names


    optional parameters :
    --------------------
    additionnalVariable : list or tuple containing elements to add another variable to the matrix (i.e. another row)
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series object
    figsize : list or tuple, size of the figure. Default = (10,5)
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # use pcaCorrelationMatrix function to create the matrix
    dfMatrix = pcaCorrelationMatrix(
        pca=pca,
        PcafeaturesNames=PcafeaturesNames,
        additionnalVariable=additionnalVariable,
    )

    # initiate plot
    fig, ax = plt.subplots(1, figsize=figsize)

    # heatmap
    sns.heatmap(
        data=dfMatrix,  # use the correlation matrix computed above
        linewidth=1,  # line between squares of the heatmap
        cmap=sns.diverging_palette(
            262, 12, as_cmap=True, center="light", n=9
        ),  # blue for anticorrelated, red for correlated
        center=0,  # no color for no correlation
        annot=True,  # displays Pearson coefficients
        fmt="0.2f",  # with 2 decimals
        ax=ax,
    )

    # change tick labels locations
    ax.tick_params(
        top=False, labeltop=True, labelbottom=False, bottom=False  # put them on top
    )
    # change tick labels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # set title
    ax.set_title("Corrélations variables / composantes principales")


def heatPcaCorrelationMatrixWid(df, X_scaled, pca, PcafeaturesNames, figsize=(10, 5)):
    """
    display a PCA correlation matrix in a Seaborn Heatmap way

    parameters :
    ----------
    pcaCorrelationMatrix : dataframe, PCA correlation matrix returned from the pcaCorrelationMatrix function

    optional parameters :
    --------------------
    figsize : list or tuple, size of the figure. Default = (10,5)
    """

    import ipywidgets as widgets

    # create a widget for choosing additionnal variable
    addVarColList = [
        col
        for col in df.columns
        if (col not in PcafeaturesNames) and (df[col].dtype.kind in "biufc")
    ]

    import ipywidgets as widgets

    widAddVar = widgets.Dropdown(
        options={col: (X_scaled, df[col]) for col in addVarColList} | {None: None},
        value=None,
        description="Additionnal variable :",
        disabled=False,
        style={"description_width": "initial"},
    )

    out = widgets.interactive_output(
        heatPcaCorrelationMatrix,
        {
            "pca": widgets.fixed(pca),
            "PcafeaturesNames": widgets.fixed(PcafeaturesNames),
            "additionnalVariable": widAddVar,
            "figsize": widgets.fixed(figsize),
        },
    )

    display(widAddVar, out)


def correlation_graph_enhanced(
    pca,
    x_y,
    PcafeaturesNames,
    normalization,
    dictPalette=None,
    figsize=(10, 9),
    additionnalVariable=None,
):
    """display correlation graph for a given pca



    Parameters :
    ----------
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    x_y : list or tuple : the couple of the factorial plan, example [1,2] for F1, F2
    PcafeaturesNames : list or tuple : list of pca features names
    normalization : string, decide what one wants to plot :
    - "loadings" - columns of V.Lambda^(1/2) - loadings
    - "principal_axis" - columns of V - principal directions/axis

    Optional parameters :
    -------------------
    dictPalette : dictionnary, with features names for keys and colors for values - default : None
    figsize : list or tuple, size of figure - default : (10,9)
    additionnalVariable : list or tuple containing elements to add another variable to the graph
        - element 0 : X_scaled, used for PCA
        - element 1 : addVarSeries, the additionnal variable pandas.Series
    """
    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    sns.set_theme()
    # Extract x and y
    x, y = x_y

    # Adjust x and y to list/array numerotation
    x = x - 1
    y = y - 1

    # compute matrix
    if normalization == "principal_axis":
        matrix = pca.components_.T  # principal axis matrix
        dfMatrix = pd.DataFrame(  # tranform matrix into a dataframe
            matrix,
            index=PcafeaturesNames,
            columns=[  # one columns for each principal vector, with its global quality of representation
                # which is the percentage of explained variance
                "F"
                + str(i + 1)
                + " ("
                + str(round(100 * pca.explained_variance_ratio_[i], 1))
                + "%)"
                for i in range(pca.n_components_)
            ],
        )
    if normalization == "loadings":  # in that we use the above function
        dfMatrix = pcaCorrelationMatrix(
            pca=pca,
            PcafeaturesNames=PcafeaturesNames,
            additionnalVariable=additionnalVariable,
        )

    # size of the image (in inches)
    fig, ax = plt.subplots(figsize=figsize)

    # For each column :
    for i, col in enumerate(dfMatrix.index.tolist()):
        x_coord = dfMatrix.iloc[i, x]
        y_coord = dfMatrix.iloc[i, y]
        ax.arrow(
            0,
            0,  # Start the arrow at the origin
            x_coord,
            y_coord,
            head_width=0.07,
            head_length=0.07,
            width=0.02,
            length_includes_head=True,  # so arrow stays inside R=1 circle
            color=dictPalette[
                col if col in dictPalette.keys() else additionnalVariable[1].name
            ]
            if dictPalette
            else None,  # if each feature has a specific color, we use it
        )

        # put name of the feature at the top of the arrow

        ax.text(
            x_coord,
            y_coord,
            col,
            horizontalalignment="left" if x_coord > 0 else "right",
            verticalalignment="bottom" if y_coord > 0 else "top",
            fontsize=10 * figsize[0] / 10,
            rotation=np.arctan(y_coord / x_coord) * 180 / np.pi,
        )

    # Display x-axis and and y-axis in dot ligns
    plt.plot([-1, 1], [0, 0], color="grey", ls="--")
    plt.plot([0, 0], [-1, 1], color="grey", ls="--")

    # Names of factorial axis, with percent of explained variance/inertia
    plt.xlabel(dfMatrix.columns.tolist()[x], fontsize=12 * figsize[0] / 10)
    plt.ylabel(dfMatrix.columns.tolist()[y], fontsize=12 * figsize[0] / 10)

    # ticks size
    plt.setp(ax.get_xticklabels(), fontsize=13 * figsize[0] / 10)
    plt.setp(ax.get_yticklabels(), fontsize=13 * figsize[0] / 10)

    plt.title(
        "Cercle des corrélations (F{} et F{})".format(x + 1, y + 1),
        fontsize=13 * figsize[0] / 10,
    )

    # circle if we use loadings
    if normalization == "loadings":
        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an), c="k")  # Add a unit circle for scale

    # Axes and display
    plt.axis("equal")  # common scale
    plt.show(block=False)


def correlation_graph_enhanced_WID(
    df,
    X_scaled,
    pca,
    PcafeaturesNames,
    dictPalette=None,
    figsize=(10, 9),
):
    """display correlation graph for a given pca, with the choice of normalization


    Parameters :
    ----------
    df - dataframe : original dataframe, from which we can extract an additional variable
    X_scaled : used for pca
    pca : sklearn.decomposition.PCA : PCA object, already fitted
    x_y : list or tuple : the couple of the factorial plan, example [1,2] for F1, F2
    PcafeaturesNames : list of strings : list of features (ie dimensions) to draw

    Optional parameters :
    -------------------
    dictPalette : dictionnary, with features names for keys and colors for values - default : None

    """
    # create ipywidgets radio buttons for normalization choice
    import ipywidgets as widgets

    widNormalization = widgets.RadioButtons(
        options=["loadings", "principal_axis"],
        value="loadings",
        description="Normalization option :",
        disabled=False,
        style={"description_width": "initial"},
    )
    # create an ipywidgets dropdown for factorial plan choice
    q = pca.n_components_  # store the numbers of components
    from itertools import combinations  # import compbinations tool

    factPlanList = list(
        combinations([f + 1 for f in range(q)], 2)
    )  # create a list of all factorial plans

    def myOrder(elem):  # create a function to sort the list of factorial plans
        return elem[1] - elem[0]

    factPlanList.sort(key=myOrder)

    factPlanList = [
        (str(plan), plan) for plan in factPlanList
    ]  #  change format for widgets compatibility

    widFactorialPlan = widgets.Dropdown(
        options=factPlanList,
        value=factPlanList[0][1],
        description="Factorial plan :",
        disabled=False,
        style={"description_width": "initial"},
    )

    # create a widget for choosing additionnal variable
    addVarColList = [
        col
        for col in df.select_dtypes("float64").columns
        if col not in PcafeaturesNames
    ]
    widAddVar = widgets.Dropdown(
        options={col: (X_scaled, df[col]) for col in addVarColList} | {None: None},
        value=None,
        description="Additionnal variable :",
        disabled=False,
        style={"description_width": "initial"},
    )

    ui = widgets.HBox([widFactorialPlan, widNormalization, widAddVar])

    out = widgets.interactive_output(
        correlation_graph_enhanced,
        {
            "pca": widgets.fixed(pca),
            "x_y": widFactorialPlan,
            "PcafeaturesNames": widgets.fixed(PcafeaturesNames),
            "normalization": widNormalization,
            "dictPalette": widgets.fixed(dictPalette),
            "figsize": widgets.fixed(figsize),
            "additionnalVariable": widAddVar,
        },
    )

    display(ui, out)


def bestDtype(series):
    """
    returns the most memory efficient dtype for a given Series

    parameters :
    ------------
    series : series from a dataframe

    returns :
    ---------
    bestDtype : dtype
    """
    # imports
    import sys
    import pandas as pd
    import gc

    # create a copy()
    s = series.copy()

    # initiate bestDtype with s dtype
    bestDtype = s.dtype

    # initiate a scalar which will contain the min memory
    bestMemory = sys.getsizeof(s)

    # return "cat" or "datetime" if dtype is of kind 'O'
    if s.dtype.kind == "O":
        # return 'datetime64[ns]' if dates are detected
        if s.str.match(r"\d{4}-\d{2}-\d{2} \d{2}\:\d{2}\:\d{2}").all(axis=0):
            bestDtype = "datetime64[ns]"
        else:
            bestDtype = "category"

    # for numericals
    else:
        # test several downcasts
        for typ in ["unsigned", "signed", "float"]:
            sDC = pd.to_numeric(s, downcast=typ)
            # if downcasted Series is different, continue
            if (s == sDC).all() == False:
                continue
            # get memory
            mem = sys.getsizeof(sDC)
            # if best, update bestDtype and bestMemory
            if mem < bestMemory:
                bestMemory = mem
                bestDtype = sDC.dtype
            del sDC
            gc.collect()

    del s
    gc.collect()
    return bestDtype


def plotCatFeatureVsTarget(
    df,
    catFeatureName,
    targetFeatureName,
    targetValues=None,
    includeCatFeatureNan=True,
    includeTargetNan=False,
    palette=None,
):
    """
    plot the distribution of a categorical feature AND the percentage of target values per category on another graph.

    parameters :
    ------------
    df - DataFrame
    catFeatureName - string : name of the categorical feature
    targetFeatureName - string : name of the target
    targetValues - list or str/int/float or None : target value(s) to consider in the "percentage" graph.
                            By default : None  (use of all Target unique values)
    includeCatFeatureNan - bool : Whether or not to include the categorical feature missing values as a category.
                            By default : True
    includeTargetNan - bool : Whether or not to include the Target missing values as a category.
                            By default : False
    palette - dict : target names as keys, colors as values
                            By default : None

    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import seaborn as sns

    # create a copy
    tempDf = df[[catFeatureName, targetFeatureName]].copy()

    # initiate a dataframe to store counts and percentages
    catFeatureDf = pd.DataFrame(
        tempDf[catFeatureName].value_counts(dropna=not includeCatFeatureNan)
    )

    # add a line with infos of the whole dataframe
    catFeatureDf.loc["WHOLE_DATA", "count"] = tempDf.shape[0]
    catFeatureDf = catFeatureDf.loc[
        [catFeatureDf.index[-1]] + list(catFeatureDf.index)[:-1]
    ]

    # handle targetValues
    if targetValues:
        if type(targetValues) != list:
            targetValues = [targetValues]
    else:
        targetValues = list(tempDf[targetFeatureName].dropna().unique())
        if tempDf[targetFeatureName].dtype.kind in "biufc":
            targetValues.sort()

        if includeTargetNan == True and tempDf[targetFeatureName].isna().sum() > 0:
            if tempDf[targetFeatureName].dtype.kind == "O":
                tempDf[targetFeatureName] = (
                    tempDf[targetFeatureName]
                    .astype("O")
                    .fillna("targetMissing")
                    .astype("category")
                )
            else:
                tempDf[targetFeatureName] = tempDf[targetFeatureName].fillna(
                    "targetMissing"
                )
            targetValues = targetValues + ["targetMissing"]

    if tempDf[targetFeatureName].dtype.kind in "biufc":
        targetValues.sort()

    # add percentages for each Target unique values
    for val in targetValues:
        catFeatureDf[val] = (
            tempDf.loc[tempDf[targetFeatureName] == val, catFeatureName].value_counts(
                dropna=not includeCatFeatureNan
            )
            / catFeatureDf["count"]
        )

        catFeatureDf.loc["WHOLE_DATA", val] = (
            tempDf[tempDf[targetFeatureName] == val].shape[0] / df.shape[0]
        )

    catFeatureDf = catFeatureDf.reset_index()
    catFeatureDf[catFeatureName] = catFeatureDf[catFeatureName].fillna(
        "'NaN'"
    )  # replace np.nan category with a string 'NaN'
    catFeatureDf[catFeatureName] = catFeatureDf[catFeatureName].astype(str)
    catFeatureDf = catFeatureDf.fillna(
        0
    )  # if a catfeature value was not in filtered tempDf value_counts, i.e. not present
    # in this filtered tempDf, its percentage has been filled with np.nan. Replace with 0%

    # use .melt() method for the "percentage" graph
    catFeatureDfMelt = catFeatureDf.drop(columns="count").melt(
        id_vars=catFeatureName,
        value_vars=catFeatureDf.columns[2:],
        var_name=targetFeatureName,
    )

    if len(catFeatureDf) < 10:
        # create figure with 2 axes

        # handle palette
        pal = sns.color_palette("Paired", len(catFeatureDf))  # palette
        # create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # figure and axes
        axs = axs.ravel()
        nanRate = str(round(tempDf[catFeatureName].isna().mean() * 100, 1)) + "%"
        fig.suptitle(
            catFeatureName
            + " ( "
            + nanRate
            + " NaN ) : distribution and relation with Target"
        )  # main title

        # plot distribution of the categorical feature
        sns.barplot(
            data=catFeatureDf.iloc[1:, :],
            x=catFeatureName,
            y="count",
            color="black"
            if len(targetValues) > 1
            else None,  # if use of several target values, countplot in black
            palette=None
            if len(targetValues) > 1
            else pal,  # if use of a unique() target value, use palette colors
            ax=axs[0],
        )

        # set axis parameters
        axs[0].set_ylabel("Number of observations")
        xticks = catFeatureDf.iloc[1:, :][catFeatureName].unique()
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(
            xticks,
            rotation=90 if len(catFeatureDf) > 3 else 0,
            ha="center",
            va="center_baseline",
        )
        axs[0].set_xlabel("")

        # plot percentage of Target values for each category of cat feature

        # handle palette
        if not palette:
            # if several target values : grey palette
            if len(targetValues) > 1:
                palette = sns.color_palette("Greys", len(targetValues) + 2)[2:]
                # if several target values : grey palette
            else:
                [
                    "black"
                ] + pal  # if unique target value : use pal and "black" for "WHOLE_DATA"
        else:
            palette = palette
        sns.barplot(
            data=catFeatureDfMelt,
            x=catFeatureName,
            y="value",
            hue=targetFeatureName
            if len(targetValues) > 1
            else None,  # if use of several target values, stacked plot
            ax=axs[1],
            order=None,
            palette=palette,
        )

        # set axis parameters
        axs[1].set_ylabel(
            "Percent of " + targetFeatureName + " values"
            if len(targetValues) > 1
            else "Percent of " + targetFeatureName + " = " + str(targetValues[0])
        )
        xticks = catFeatureDfMelt[catFeatureName].unique()
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(
            xticks,
            rotation=90 if len(catFeatureDf) > 3 else 0,
            ha="center",
            va="center_baseline",
        )
        axs[1].set_xlabel("")
        axs[1].set_yticks(list(axs[1].get_yticks()) + list(catFeatureDf.iloc[0, 2:]))
        axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # plot horizontal line(s) for each target value. To compare "WHOLE_DATA" percentage to each category percentage
        for i, val in enumerate(targetValues):
            axs[1].axhline(
                y=catFeatureDf.iloc[0, 2 + i],
                color=sns.color_palette("Greys", len(targetValues) + 2)[2:][i]
                if len(targetValues) > 1
                else "black",
            )

    else:
        # create figure with 2 axes
        if not palette:
            palette = sns.color_palette("Paired", len(catFeatureDf))  # palette
        fig, axs = plt.subplots(1, 2, figsize=(12, 12))  # figure and axes
        axs = axs.ravel()
        nanRate = str(round(tempDf[catFeatureName].isna().mean() * 100, 1)) + "%"
        fig.suptitle(
            catFeatureName
            + " ( "
            + nanRate
            + " NaN ) : distribution and relation with Target"
        )  # main title

        # plot distribution of the categorical feature
        sns.barplot(
            data=catFeatureDf.iloc[1:, :],
            y=catFeatureName,
            x="count",
            color="black"
            if len(targetValues) > 1
            else None,  # if use of several target values, countplot in black
            palette=None
            if len(targetValues) > 1
            else palette,  # if use of a unique() target value, use palette colors
            ax=axs[0],
        )

        # set axe parameters
        axs[0].set_xlabel("Number of observations")
        axs[0].set_ylabel("")
        axs[0].tick_params(  # change tick labels locations
            top=True, labeltop=True, labelbottom=True, bottom=True  # put them on top
        )

        # plot percentage of Target values for each category of cat feature
        sns.barplot(
            data=catFeatureDfMelt.loc[catFeatureDfMelt[catFeatureName] != "WHOLE_DATA"],
            y=catFeatureName,
            x="value",
            hue=targetFeatureName
            if len(targetValues) > 1
            else None,  # if use of several target values, stacked plot
            ax=axs[1],
            order=None,
            palette=sns.color_palette("Greys", len(targetValues) + 2)[2:]
            if len(targetValues) > 1  # if several target values : grey palette
            else palette,  # if unique target value : use palette and "black" for "WHOLE_DATA"
        )

        # set axe parameters
        axs[1].set_xlabel(
            "Percent of " + targetFeatureName + " values"
            if len(targetValues) > 1
            else "Percent of " + targetFeatureName + " = " + str(targetValues[0])
        )
        axs[1].set_ylabel("")
        axs[1].set_yticklabels("")
        axs[1].set_xticks(list(axs[1].get_xticks()) + list(catFeatureDf.iloc[0, 2:]))
        axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axs[1].tick_params(  # change tick labels locations
            top=True, labeltop=True, labelbottom=True, bottom=True  # put them on top
        )

        # plot horizontal line(s) for each target value. To compare "WHOLE_DATA" percentage to each category percentage
        for i, val in enumerate(targetValues):
            axs[1].axvline(
                x=catFeatureDf.iloc[0, 2 + i],
                color=sns.color_palette("Greys", len(targetValues) + 2)[2:][i]
                if len(targetValues) > 1
                else "black",
            )

    del tempDf, catFeatureDf, palette, targetValues, catFeatureDfMelt, nanRate


def plotNumFeatureVsTarget(
    df,
    numFeatureName,
    targetFeatureName,
    targetValues=None,
    includeTargetNan=False,
    sampleSize=None,
    palette=None,
):
    """
    plot the distribution of a numerical feature for the whole data AND for filtered data on values of the target.

    parameters :
    ------------
    df - DataFrame
    numFeatureName - string : name of the numerical feature
    targetFeatureName - string : name of the target
    targetValues - list or str/int/float or None : target value(s) to consider in the "percentage" graph.
                            By default : None  (use of all Target unique values)
    includeTargetNan : bool : Whether or not to include the Target missing values as a category.
                            By default : False,
    sampleSize - int or None : size of the df sample ( for faster plotting)
                            By default : None  (no sampling)
    palette - dict : target names as keys, colors as values
                            By default : None

    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # create a copy
    if sampleSize:
        tempDf = df[[numFeatureName, targetFeatureName]].copy().sample(sampleSize)
    else:
        tempDf = df[[numFeatureName, targetFeatureName]].copy()

    # handle targetValues
    if targetValues:
        if type(targetValues) != list:
            targetValues = [targetValues]
    else:
        targetValues = list(tempDf[targetFeatureName].dropna().unique())
        if tempDf[targetFeatureName].dtype.kind in "biufc":
            targetValues.sort()

        if includeTargetNan == True:
            if tempDf[targetFeatureName].dtype.kind == "O":
                tempDf[targetFeatureName] = (
                    tempDf[targetFeatureName]
                    .astype("O")
                    .fillna("targetMissing")
                    .astype("category")
                )
            else:
                tempDf[targetFeatureName] = tempDf[targetFeatureName].fillna(
                    "targetMissing"
                )
            targetValues = targetValues + ["targetMissing"]

    # filtered df on target values
    filteredDf = []
    for val in targetValues:
        if type(val) == float and np.isnan(val):  # handle NaN
            filteredDf.append(tempDf.loc[tempDf[targetFeatureName].isna()])
        else:
            filteredDf.append(tempDf.loc[tempDf[targetFeatureName] == val])

    # plot

    # create figure with 2 axes
    if not palette:
        palette = sns.color_palette("Paired")  # palette
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # figure and axes
    axs = axs.ravel()
    nanRate = str(round(tempDf[numFeatureName].isna().mean() * 100, 1)) + "%"
    fig.suptitle(
        numFeatureName
        + " ( "
        + nanRate
        + " NaN ) : distribution and relation with Target"
    )  # main title

    sns.histplot(
        data=tempDf,
        x=numFeatureName,
        stat="density",
        bins=100,
        kde=True,
        ax=axs[0],
        ec=None,
    )

    for val, df in zip(targetValues, filteredDf):
        sns.kdeplot(
            data=df,
            x=numFeatureName,
            ax=axs[1],
            label=targetFeatureName + " = " + str(val),
            legend=True,
            cut=0,
            fill=True,
            color=palette[val],
        )

    axs[1].legend()

    del tempDf, filteredDf, palette, targetValues, nanRate


def plotFeatureVsTargetWID(
    df,
    targetFeatureName,
    targetValsForCat=None,
    targetValsForNum=None,
    includeCatFeatureNan=True,
    includeTargetNan=False,
    palette=None,
):
    """
    plot the distribution of a numerical or categorical feature for the whole data AND for filtered data on values of the target, which options to select :
                - the feature
                - the number of unique values under which a Numerical feature should be considered as Categorical

    parameters :
    ------------
    df - DataFrame
    targetFeatureName - string : name of the target
    targetValsForCat - list or str/int/float or None : target value(s) to consider in the "percentage" graph.
                            By default : None  (use of all Target unique values)
    targetValsForNum - list or str/int/float or None : target value(s) to consider in the "percentage" graph.
                            By default : None  (use of all Target unique values)
    includeCatFeatureNan - bool : Whether or not to include the categorical feature missing values as a category.
                            By default : True
    includeTargetNan - bool : Whether or not to include the Target missing values as a category.
                            By default : False
    palette - dict : target names as keys, colors as values
                            By default : None

    """
    import ipywidgets as widgets
    import pandas as pd
    import numpy as np

    # widget - categorical variable selection
    colList = [col for col in df.columns if col != targetFeatureName]

    widCol = widgets.Dropdown(
        options=colList,
        value=colList[0],
        description="Which column :",
        disabled=False,
        style={"description_width": "initial"},
    )

    widCatOrNumThreshold = widgets.IntSlider(
        value=50,
        min=10,
        max=100,
        step=10,
        description="Nb of unique values below which a num' feature is considered cat' :",
        style={"description_width": "initial"},
        disabled=False,
        layout=widgets.Layout(width="75%"),
    )

    def whichPlot(col, threshold):
        if (df[col].dtype.kind in "O") or (len(df[col].unique()) <= threshold):
            plotCatFeatureVsTarget(
                df=df,
                catFeatureName=col,
                targetFeatureName=targetFeatureName,
                targetValues=targetValsForCat,
                includeCatFeatureNan=includeCatFeatureNan,
                includeTargetNan=includeTargetNan,
                palette=palette,
            )
        else:
            widSampleSize = widgets.BoundedIntText(
                value=10000,
                min=0,
                max=len(df),
                step=1,
                description="Sample size : ",
                disabled=False,
            )

            outSub = widgets.interactive_output(
                plotNumFeatureVsTarget,
                {
                    "df": widgets.fixed(df),
                    "numFeatureName": widgets.fixed(col),
                    "targetFeatureName": widgets.fixed(targetFeatureName),
                    "targetValues": widgets.fixed(targetValsForNum),
                    "includeTargetNan": widgets.fixed(includeTargetNan),
                    "sampleSize": widSampleSize,
                    "palette": widgets.fixed(palette),
                },
            )
            display(widSampleSize, outSub)

    out = widgets.interactive_output(
        whichPlot, {"col": widCol, "threshold": widCatOrNumThreshold}
    )
    display(widCol, widCatOrNumThreshold, out)


def removeAccent(text):
    """
    remove accent(s) from a given string
    parameters :
    ------------
    text - string

    returns :
    ---------
    textWithoutAccent - string : same text without accent
    """
    # imports
    import unicodedata

    # remove accent
    textWithoutAccent = (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("utf-8", "ignore")
    )

    return textWithoutAccent


def myMode(x, missing="missing_value"):
    """
    parameter :
    -----------
    x - Series
    missing - scalar : what should be returned if the Series is empty

    return :
    --------
    myMode - scalar : the mode of the Serie (or one of them if several)
    """

    # handle empty col
    if x.isna().all(axis=0):
        myMode = "missing"
    # compute the mode
    else:
        myMode = x.value_counts().index[0]
    return myMode


def nanTab(df, digits=2):
    """
    displays a dataframe with the NaN rate of each column of a given dataframe
    parameter :
    -----------
    df - dataframe
    digits - int : number of decimals to use when rounding the percentage. By default : 2

    output :
    -------
    displays the NaN rate tab

    """
    # extract the nan rates
    nanTab = df.isna().mean()

    # convert Series to Dataframe
    nanTab = nanTab.to_frame(name="NaN_rate")

    # percentages
    nanTab = nanTab * 100

    # round
    nanTab = round(nanTab, digits)

    # cast to str and add "%"  character
    nanTab = nanTab.astype(str) + " %"

    # display
    display(nanTab)


def displayHeatMapCorrMatrix(df, method="pearson", figsize=(10, 10)):
    """
    Given a numerical dataframe, compute and displays a correlation matrix, with seaborn heatmap

    parameters :
    ------------
    df - dataframe : ONLY NUMERICAL
    method - string : "pearson", "kendall" or "spearman". By default : "pearson"
    figsize - tuple of int. By default : (10,10)

    returns :
    ---------
    matplotlib Axes

    """
    # imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # compute correlation matrix
    corr = df.corr(method=method)
    # only below diag
    maskTri = np.triu(np.ones_like(corr, dtype=bool), k=0)
    corr = corr.mask(maskTri)
    # in percentages
    corr = corr * 100

    # initiate figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)

    # heatmap
    sns.heatmap(
        data=corr,
        linewidths=1,  # line between squares of the heatmap
        cmap=sns.diverging_palette(
            262, 12, as_cmap=True, center="light", n=9
        ),  # blue for anticorrelated, red for correlated
        center=0,  # no color for no correlation,
        annot=True,  # displays Pearson coefficients
        fmt="0.0f",  # with 0 decimals
        ax=ax,
        annot_kws={"fontsize": figsize[0]},  # set fontsize in cells
        square=True,  # square cells
        cbar=False,  # hide color bar
    )

    # add ticks
    ax.tick_params(
        axis="both",
        left=True,
        bottom=True,
    )
    # rotate xtick labels
    ax.set_xticklabels(ax.get_xticklabels(), ha="right", va="top", rotation=45)

    # title
    ax.set_title("Correlation matrix - " + method)

    plt.show()


def displayHeatMapCorrMatrixWID(df, figsize=(10, 10)):
    """
    Given a numerical dataframe, compute and displays a correlation matrix, with seaborn heatmap
    With an option to select the method between "pearson", "kendall" or "spearman"

    parameters :
    ------------
    df - dataframe : ONLY NUMERICAL
    figsize - tuple of int. By default : (10,10)

    returns :
    ---------
    matplotlib Axes

    """

    # imports
    import ipywidgets as widgets

    # method selection widget
    widMethod = widgets.RadioButtons(
        options=["pearson", "kendall", "spearman"],
        value="pearson",
        description="Method : ",
    )

    # interactive output using displayHeatMapCorrMatrix function
    out = widgets.interactive_output(
        displayHeatMapCorrMatrix,
        {
            "df": widgets.fixed(df),
            "method": widMethod,
            "figsize": widgets.fixed(figsize),
        },
    )

    # ui
    display(widMethod, out)


def myscatter3D(dataframe, sample=None, palette=None, title=None):
    """
    use plotly scatter3D to plot on a 3D or 4D dataframe which has already been prepared

    parameters :
    ------------
    dataframe - dataframe : 3D or 4D dataframe
                           - with the first 3 columns as x, y, z. All numericals
                           - the 4th column as hue (optionnal). dtype must be "category"
    sample - int : the number of samples, if ones wants to sample before plotting for better readability. By default : None (no sampling)
    palette - dict : 4th column categories as keys, colors as values
    title - string : title text. If None, will create a title based on the dataframe columns

    outputs :
    ---------
    displays a plotly figure, 3D scatter
    """

    # imports
    import pandas as pd
    import plotly.express as px

    # extract x, y, z and hue
    df = dataframe.copy()
    myX = df.columns[0]
    myY = df.columns[1]
    myZ = df.columns[2]

    # handle 3D or 4D, for hue
    if df.shape[1] == 4:
        # get columns name
        myHue = df.columns[3]
        # get its categories and put them as a list in a dictionary
        catList = df[myHue].cat.categories.to_list()
        myCategory_orders = {myHue: catList}

    else:
        myHue = None
        myCategory_orders = None

    # handle sampling
    if sample:
        df = df.sample(sample, random_state=16)

    # create the figure
    fig = px.scatter_3d(
        data_frame=df,
        x=myX,
        y=myY,
        z=myZ,
        opacity=0.8,
        width=1000,
        height=1000,
        color=myHue,
        color_discrete_map=palette,
        category_orders=myCategory_orders,
        template="plotly_white",
    )

    # set title
    if title:
        myTitle = title
    else:
        myTitle = "3D scatter plot :\n" + myX + " by " + myY + " by " + myZ
        if myHue:
            myTitle = myTitle + "<br><br> hue on " + myHue
    fig.update_layout(
        title_text=myTitle,
    )

    fig.show()


def scaling_and_Kmeans(dataframe, n_clusters, scaler=None, random_state=None):
    """
    perform a scaling then a kmeans on a given dataframe

    parameters :
    ------------
    dataframe - dataframe
    n_clusters - int
    scaler - string : among "std", "rob" and None. By default, None (no scaling)
    random_state - int : random_state parameter for KMeans algorithm. By default, None

    returns :
    ---------
    df - dataframe : same one, with a new column "cluster"
    df_scaled - dataframe : dito, but scaled (optionnal)

    """

    # imports
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    import pandas as pd

    # make a copy
    df = dataframe.copy()

    # put scalers in a disctionary (for pipeline purpose)
    scalerDict = {"std": StandardScaler(), "rob": RobustScaler(), None: "passthrough"}

    # kmeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=15, random_state=random_state)

    # pipeline
    pipe = Pipeline([("scaler", scalerDict[scaler]), ("kmeans", kmeans)])

    # predict
    clusters = pd.Categorical(pipe.fit_predict(df))

    # if wanted, create df_scaled
    if scaler:
        # scale
        df_scaled = pd.DataFrame(
            scalerDict[scaler].fit_transform(df), columns=df.columns
        )
        # add "clusters"
        df_scaled["clusters"] = clusters

    # add "clusters" to df
    df["clusters"] = clusters

    # returns
    if scaler:
        return df, df_scaled
    else:
        return df


def clusters_plot2D(dataframe, palette, sample=None, ax=None, equal=True):
    """
    plot a seaborn scatterplot of 2D dataframe with hue on clusters

    parameters :
    ------------
    dataframe - dataframe : a 2D+1D dataframe
                                - first 2 columns : numericals
                                - last column : clusters
    palette - dict : clusters names as keys, colors as values
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)
    ax : matplotlib axes. By default : None
    equal - bool : wether or not to display with equal aspect ratio


    output :
    --------
    displays a 2D scatter, with hue on clusters

    """

    # imports
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # make a copy
    df = dataframe.copy()

    # sample
    if sample:
        df = df.sample(sample)

    # plot
    sns.scatterplot(
        data=df,
        x=df.columns[0],
        y=df.columns[1],
        hue=df.columns[2],
        palette=palette,
        ax=ax,
    )

    # equal scale
    if equal == True:
        ax = plt.gca()
        ax.axis("equal")

    plt.title("Clustering - 2D scatter plot")

    if not ax:
        plt.show()


def clusters_plot3D(dataframe, palette, sample=None):
    """
    plot a plotly 3D scatter (using myscatter3D function)

    parameters :
    ------------
    dataframe - dataframe : a 3D+1D dataframe
                                - first 3 columns : numericals
                                - last column : clusters
    palette - dict : clusters names as keys, colors as values
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)


    output :
    --------
    displays a 3D scatter, with hue on clusters

    """

    # imports
    import pandas as pd
    import seaborn as sns

    # make a copy
    df = dataframe.copy()

    # plot
    myscatter3D(
        dataframe=df,
        sample=sample,
        palette=palette,
        title="Clustering - 3D scatter plot",
    )


def plot_scaling_and_Kmeans_2D(
    dataframe,
    n_clusters,
    palette,
    scaler=None,
    random_state=None,
    sample=None,
    whichDf="initial",
):
    """
    combine "scaling_and_Kmeans" function and "clusters_plot2D" function to :
                - perform a kmeans clustering on a 2D dataframe
                - plot 2D scatter with hue on clusters

    parameters :
    ------------
    dataframe - dataframe : 2 dimensions, only numericals
    n_clusters - int
    palette - dict : clusters names as keys, colors as values
    scaler - string : among "std", "rob" and None. By default, None (no scaling)
    random_state - int : random_state parameter for KMeans algorithm. By default, None
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)
    whichDf - string : among "initial" and "scaled" (only if scaler is not None). hence, by default : "initial"

    output :
    --------
    displays a 2D scatter, with hue on clusters

    """

    # handle scaler parameters and use "scaling_and_Kmeans" function
    if scaler:
        df, df_scaled = scaling_and_Kmeans(
            dataframe=dataframe,
            n_clusters=n_clusters,
            scaler=scaler,
            random_state=random_state,
        )
    else:
        df = scaling_and_Kmeans(
            dataframe=dataframe, n_clusters=n_clusters, random_state=random_state
        )

    # select the dataframe used for plotting
    if whichDf == "initial":
        plotDf = df
    elif whichDf == "scaled":
        plotDf = df_scaled
    # use clusters_plot2D
    clusters_plot2D(dataframe=plotDf, palette=palette, sample=sample)


def plot_scaling_and_Kmeans_2D_WID(dataframe, palette):
    """
    use "plot_scaling_and_Kmeans_2D" to perform a KMeans on a given 2D numerical dataframe.
    with options to choose the scaler, the number of clusters, samples, etc.

    parameters :
    ------------
    dataframe - dataframe : 2D numericals
    palette - dict : clusters names as keys, colors as values

    returns :
    ---------
    displays a 2D scatter with hue on clusters

    """

    # imports
    import ipywidgets as wid

    # create a widget for scaling
    widScaler = wid.RadioButtons(
        options={"No scaling": None, "Standard scaler": "std", "Robust scaler": "rob"},
        value="std",
        description="Scaler : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for random_state
    widRS = wid.Dropdown(
        options=[None] + [i + 1 for i in range(10)],
        value=None,
        description="Random_state : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for the number of clusters
    widClusterNumber = wid.BoundedIntText(
        min=1,
        value=1,
        description="nb of cluster : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for sampling (for plotting purpose)
    widSample = wid.BoundedIntText(
        min=1,
        max=len(dataframe),
        value=20000,
        description="nb of samples : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget to choose wether to use original df or scaled one for plotting
    widDf = wid.RadioButtons(
        options=["initial", "scaled"],
        value="initial",
        description="for plotting, which df : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # handle widScaler changes
    def handleScalerChange(change):
        if change.new == None:
            widDf.options = ["initial"]
        if change.new != None:
            widDf.options = ["initial", "scaled"]
            widDf.value = "initial"

    widScaler.observe(handleScalerChange, "value")

    ui1 = wid.VBox([widClusterNumber, widScaler])
    ui2 = wid.VBox([widSample, widDf])
    ui = wid.HBox([ui1, widRS, ui2], layout=wid.Layout(justify_content="space-around"))
    # use interactive_output
    out = wid.interactive_output(
        plot_scaling_and_Kmeans_2D,
        {
            "dataframe": wid.fixed(dataframe),
            "n_clusters": widClusterNumber,
            "palette": wid.fixed(palette),
            "scaler": widScaler,
            "random_state": widRS,
            "sample": widSample,
            "whichDf": widDf,
        },
    )

    display(ui, out)


def plot_scaling_and_Kmeans_3D(
    dataframe,
    n_clusters,
    palette,
    scaler=None,
    random_state=None,
    sample=None,
    whichDf="initial",
):
    """
    combine "scaling_and_Kmeans" function and "clusters_plot3D" function to :
                - perform a kmeans clustering on a 3D dataframe
                - plot 3D scatter with hue on clusters

    parameters :
    ------------
    dataframe - dataframe : 3 dimensions, only numericals
    n_clusters - int
    palette - dict : clusters names as keys, colors as values
    scaler - string : among "std", "rob" and None. By default, None (no scaling)
    random_state - int : random_state parameter for KMeans algorithm. By default, None
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)
    whichDf - string : among "initial" and "scaled" (only if scaler is not None). hence, by default : "initial"

    output :
    --------
    displays a 3D scatter, with hue on clusters

    """

    # handle scaler parameters and use "scaling_and_Kmeans" function
    if scaler:
        df, df_scaled = scaling_and_Kmeans(
            dataframe=dataframe,
            n_clusters=n_clusters,
            scaler=scaler,
            random_state=random_state,
        )
    else:
        df = scaling_and_Kmeans(
            dataframe=dataframe, n_clusters=n_clusters, random_state=random_state
        )

    # select the dataframe used for plotting
    if whichDf == "initial":
        plotDf = df
    elif whichDf == "scaled":
        plotDf = df_scaled
    # use clusters_plot3D
    clusters_plot3D(dataframe=plotDf, palette=palette, sample=sample)


def plot_scaling_and_Kmeans_3D_WID(dataframe, palette):
    """
    use "plot_scaling_and_Kmeans_3D" to perform a KMeans on a given 3D numerical dataframe.
    with options to choose the scaler, the number of clusters, samples, etc.

    parameters :
    ------------
    dataframe - dataframe : 3D numericals
    palette - dict : clusters names as keys, colors as values

    returns :
    ---------
    displays a 3D scatter with hue on clusters

    """

    # imports
    import ipywidgets as wid

    # create a widget for scaling
    widScaler = wid.RadioButtons(
        options={"No scaling": None, "Standard scaler": "std", "Robust scaler": "rob"},
        value="std",
        description="Scaler : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for random_state
    widRS = wid.Dropdown(
        options=[None] + [i + 1 for i in range(10)],
        value=None,
        description="Random_state : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for the number of clusters
    widClusterNumber = wid.BoundedIntText(
        min=1,
        value=1,
        description="nb of cluster : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for sampling (for plotting purpose)
    widSample = wid.BoundedIntText(
        min=1,
        max=len(dataframe),
        value=20000,
        description="nb of samples : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget to choose wether to use original df or scaled one for plotting
    widDf = wid.RadioButtons(
        options=["initial"],
        value="initial",
        description="for plotting, which df : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # handle widScaler changes
    def handleScalerChange(change):
        if change.new == None:
            widDf.options = ["initial"]
        if change.new != None:
            widDf.options = ["initial", "scaled"]
            widDf.value = "initial"

    widScaler.observe(handleScalerChange, "value")

    ui1 = wid.VBox([widClusterNumber, widScaler])
    ui2 = wid.VBox([widSample, widDf])
    ui = wid.HBox([ui1, widRS, ui2], layout=wid.Layout(justify_content="space-around"))
    # use interactive_output
    out = wid.interactive_output(
        plot_scaling_and_Kmeans_3D,
        {
            "dataframe": wid.fixed(dataframe),
            "n_clusters": widClusterNumber,
            "palette": wid.fixed(palette),
            "scaler": widScaler,
            "random_state": widRS,
            "sample": widSample,
            "whichDf": widDf,
        },
    )

    display(ui, out)


def scaling_and_DBSCAN(dataframe, eps, min_samples, scaler=None):
    """
    perform a scaling then a DBSCAN on a given dataframe

    parameters :
    ------------
    dataframe - dataframe
    eps - float : The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples - int : The number of samples in a neighborhood for a point to be considered as a core point.
    scaler - string : among "std", "rob" and None. By default, None (no scaling)

    returns :
    ---------
    df - dataframe : same one, with a new column "cluster"
    df_scaled - dataframe : dito, but scaled (optionnal)

    """

    # imports
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.cluster import DBSCAN
    from sklearn.pipeline import Pipeline
    import pandas as pd

    # make a copy
    df = dataframe.copy()

    # put scalers in a disctionary (for pipeline purpose)
    scalerDict = {"std": StandardScaler(), "rob": RobustScaler(), None: "passthrough"}

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=14)

    # pipeline
    pipe = Pipeline([("scaler", scalerDict[scaler]), ("db", db)])

    # predict
    clusters = pd.Categorical(pipe.fit_predict(df))

    # if wanted, create df_scaled
    if scaler:
        # scale
        df_scaled = pd.DataFrame(
            scalerDict[scaler].fit_transform(df), columns=df.columns
        )
        # add "clusters"
        df_scaled["clusters"] = clusters

    # add "clusters" to df
    df["clusters"] = clusters

    # returns
    if scaler:
        return df, df_scaled
    else:
        return df


def plot_scaling_and_DBSCAN_2D(
    dataframe, eps, min_samples, palette, scaler=None, sample=None, whichDf="initial"
):
    """
    combine "scaling_and_DBSCAN" function and "clusters_plot2D" function to :
                - perform a DBSCAN clustering on a 2D dataframe
                - plot 2D scatter with hue on clusters

    parameters :
    ------------
    dataframe - dataframe : 2 dimensions, only numericals
    eps - float : The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples - int : The number of samples in a neighborhood for a point to be considered as a core point.
    palette - dict : clusters names as keys, colors as values
    scaler - string : among "std", "rob" and None. By default, None (no scaling)
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)
    whichDf - string : among "initial" and "scaled" (only if scaler is not None). hence, by default : "initial"

    output :
    --------
    displays a 2D scatter, with hue on clusters

    """

    # handle scaler parameters and use "scaling_and_Kmeans" function
    if scaler:
        df, df_scaled = scaling_and_DBSCAN(
            dataframe=dataframe, eps=eps, min_samples=min_samples, scaler=scaler
        )
    else:
        df = scaling_and_DBSCAN(dataframe=dataframe, eps=eps, min_samples=min_samples)

    # select the dataframe used for plotting
    if whichDf == "initial":
        plotDf = df
    elif whichDf == "scaled":
        plotDf = df_scaled
    # use clusters_plot2D
    clusters_plot2D(dataframe=plotDf, palette=palette, sample=sample)


def plot_scaling_and_DBSCAN_2D_WID(dataframe, palette):
    """
    use "plot_scaling_and_DBSCAN_2D" to perform a DBSCAN on a given 2D numerical dataframe.
    with options to choose the scaler, eps, min_samples, drawn samples, etc.

    parameters :
    ------------
    dataframe - dataframe : 2D numericals
    palette - dict : clusters names as keys, colors as values

    returns :
    ---------
    displays a 2D scatter with hue on clusters

    """

    # imports
    import ipywidgets as wid

    # create a widget for scaling
    widScaler = wid.RadioButtons(
        options={"No scaling": None, "Standard scaler": "std", "Robust scaler": "rob"},
        value="std",
        description="Scaler : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for the number of clusters
    widEps = wid.BoundedFloatText(
        min=0,
        value=0.15,
        step=0.01,
        description="epsilon distance : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for the number of clusters
    widMinSamples = wid.BoundedIntText(
        min=1,
        max=80000,
        value=50,
        description="min_samples for core point : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for sampling (for plotting purpose)
    widSample = wid.BoundedIntText(
        min=1,
        max=len(dataframe),
        value=20000,
        description="nb of drawn samples : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget to choose wether to use original df or scaled one for plotting
    widDf = wid.RadioButtons(
        options=["initial", "scaled"],
        value="scaled",
        description="for plotting, which df : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # handle widScaler changes
    def handleScalerChange(change):
        if change.new == None:
            widDf.options = ["initial"]
        if change.new != None:
            widDf.options = ["initial", "scaled"]
            widDf.value = "initial"

    widScaler.observe(handleScalerChange, "value")

    ui1 = wid.VBox([widEps, widMinSamples])
    ui2 = wid.VBox([widSample, widDf])
    ui = wid.HBox(
        [widScaler, ui1, ui2], layout=wid.Layout(justify_content="space-around")
    )
    # use interactive_output
    out = wid.interactive_output(
        plot_scaling_and_DBSCAN_2D,
        {
            "dataframe": wid.fixed(dataframe),
            "eps": widEps,
            "min_samples": widMinSamples,
            "palette": wid.fixed(palette),
            "scaler": widScaler,
            "sample": widSample,
            "whichDf": widDf,
        },
    )

    display(ui, out)


def groupByCluster(df, clusters, func="mean"):
    """
    use pandas groupby function to compute agregates over clusters given by a series

    parameters :
    ------------
    df - dataframe : contains features used for clustering
    clusters - series : cluster of each observation
    func - str or function : used in the .groupby.agg method. By default : "mean"

    return :
    --------
    dfByClusters - dataframe : grouped by dataframe
    """

    # imports
    import pandas as pd
    import gc

    # create a copy()
    dfClusters = df.copy()

    # add clusters as a column
    dfClusters["cluster"] = clusters

    # aggregate
    dfByClusters = dfClusters.groupby(by="cluster", observed=True).agg(func)

    # handle columns
    dfByClusters.columns = [col for col in dfByClusters.columns]

    # reset index
    dfByClusters.reset_index(inplace=True)

    del dfClusters
    gc.collect

    return dfByClusters


def melt_groupByCluster(dfByClusters):
    """
    given the output of the groupByCluster function, use pandas .melt with
        "cluster" as the id variable
        "R", "F", "M" as value vairiables

    parameters :
    ------------
    dfByClusters - dataframe : output of the groupByCluster function

    return :
    --------
    dfByClusters_melt - dataframe : unpivoted dfByClusters
    """

    # imports
    import pandas as pd

    # prepare parameters for .melt
    id_vars = "cluster"
    value_vars = [col for col in dfByClusters.columns if col != "cluster"]

    # melt
    dfByClusters_melt = dfByClusters.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=None,
        value_name="value",
    )

    return dfByClusters_melt


def clusteringRadarPlot(df, df_scaled, clusters, func="mean", palette=None, title=None):
    """
    use groupByCluster, melt_groupByCluster and plotly express line_polar function to display in a Radar Plot a measure of each feature by cluster

    parameters :
    ------------
    df - dataframe : contains features used for clustering
    df_scaled - dataframe : same one, scaled with standard scaler
    clusters - series : cluster of each observation
    func - str or function : used in the .groupby.agg method. By default : "mean"
    palette - dict : clusters names as keys, colors as values. By default : None
    title - string : plot title. By default : None

    output :
    ---------
    a plotly express line_polar plot

    """

    # imports
    import plotly.express as px
    import gc

    # use groupByCluster function
    dfByClusters = groupByCluster(df, clusters, func)
    df_scaledByClusters = groupByCluster(df_scaled, clusters, func)

    # use melt_groupByCluster function
    dfByClusters_melt = melt_groupByCluster(dfByClusters)
    df_scaledByClusters_melt = melt_groupByCluster(df_scaledByClusters)

    # add scaled values to dfByClusters_melt
    dfByClusters_melt["value_scaled"] = df_scaledByClusters_melt["value"]
    del df_scaledByClusters_melt
    gc.collect()

    # use .line_polar function to draw a radar plot
    fig = px.line_polar(
        dfByClusters_melt,
        r="value_scaled",  # use scaled values to have a homogeneous plot
        theta="variable",
        color="cluster",
        template="plotly_white",
        line_close=True,
        color_discrete_map=palette,
        color_discrete_sequence=None,
        hover_name="variable",
        hover_data={
            "cluster": False,
            "variable": False,
            "value": ":.2f",  # use non scaled values to have the actual value in hover
            "value_scaled": False,
        },
    )

    # set size, title and remove tick labels
    fig.update_layout(
        height=500, title_text=title, polar=dict(radialaxis=dict(showticklabels=False))
    )

    fig.show()


def exploreClusters(df, featureName, clusterName, clusteringModelName, palette):
    """
    given a dataframe containing features and a cluster columns, draw two distributions of a feature :
            - a box plot
            - a box with hue on cluster
            - a "100% stacked" bar plot, with hue on cluster
            - a pie plot with proportions of each cluster

    parameters :
    ------------
    df - dataframe : containing features and a cluster column
    featureName - string : name of the feature
    clusterName - string : name of the columns containing cluster names
    clusteringModelName - string : name of the model used for clustering
    palette - dict : clusters names as keys, colors as values

    output :
    --------
    displays a boxplot and a '100% stacked' bar plot, by cluster

    """

    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick  # for format axis in percent %
    import matplotlib.gridspec as gridspec
    import pandas as pd

    # create fig and both axes, a smaller one for the boxplot
    # fig, (ax_box, ax_cluster) = plt.subplots(2,1, sharex=True,height_ratios=(1,4), figsize=(14,6))
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(10, 10)
    ax_box = fig.add_subplot(gs[0:1, :8])
    ax_box_hue = fig.add_subplot(gs[2:3, :8])
    ax_100Stacked = fig.add_subplot(gs[4:, :8])
    ax_percentCluster = fig.add_subplot(gs[:6, 8:])

    ## box plot
    sns.boxplot(data=df, x=featureName, ax=ax_box, boxprops={"alpha": 0.7}, color="k")
    sns.boxplot(
        data=df,
        x=featureName,
        ax=ax_box_hue,
        hue=clusterName,
        palette=palette,
        boxprops={"alpha": 0.7},
    )
    # remove box_plot and box_plot_hue x ticks and label
    ax_box.xaxis.set_tick_params(labelbottom=False)
    ax_box.set_xlabel("")
    ax_box_hue.xaxis.set_tick_params(labelbottom=False)
    ax_box_hue.set_xlabel("")

    ## '100% stacked' bar plot
    myBins = min([60, df[featureName].nunique() * 2 - 1])
    sns.histplot(
        data=df,
        x=featureName,
        bins=myBins,
        ax=ax_100Stacked,
        hue=clusterName,
        element="bars",
        multiple="fill",
        palette=palette,
    )
    # change legend
    legend_style = dict(ncols=1, borderpad=5, frameon=False, fontsize=10)
    sns.move_legend(
        ax_100Stacked, "upper center", bbox_to_anchor=(1.13, 0.8), **legend_style
    )
    ax_box_hue.get_legend().remove()
    # put y axis in percent
    ax_100Stacked.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # change xticks if low number of unique values
    if myBins != 60:
        ax_100Stacked.set_xticks(
            ticks=df[featureName].unique(),
        )

    ## clusters percentages pie chart
    cluster_counts = df[clusterName].value_counts(sort=False)
    ax_percentCluster.pie(
        cluster_counts,
        colors=palette.values(),
        autopct="%1.0f%%",
        pctdistance=0.8,
        wedgeprops={"alpha": 0.7},
        textprops={"color": "w", "size": 8, "weight": "bold"},
    )
    # remove percentCluster ticks and labels
    ax_percentCluster.xaxis.set_major_locator(mtick.NullLocator())
    ax_percentCluster.yaxis.set_major_locator(mtick.NullLocator())

    # titles
    fig.suptitle(
        clusteringModelName
        + " - "
        + "feature '"
        + featureName
        + "' distribution, by cluster"
    )
    ax_box.set_title("box plot :", loc="left")
    ax_box_hue.set_title("box plot, by cluster :", loc="left")
    ax_100Stacked.set_title("'100% stacked' bar plot, by cluster :", loc="left")

    plt.show()


def exploreClustersWID(df, clusterName, clusteringModelName, palette):
    """
    use the exploreClusters function, with choice of the featureName

    parameters :
    ------------
    df - dataframe : containing featrures and a cluster column
    clusterName - string : name of the columns containing cluster names
    clusteringModelName - string : name of the model used for clustering
    palette - dict : clusters names as keys, colors as values

    output :
    --------
    displays an ipywidgets RadioButton to select the feature and a boxplot and a '100% stacked' bar plot, by cluster

    """

    # imports
    import ipywidgets as wid

    # widget to select the feature
    widFeature = wid.RadioButtons(
        options=[col for col in df.columns if col != clusterName],
        description="Feature : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # use exploreClusters function with .interactive_output()
    out = wid.interactive_output(
        exploreClusters,
        {
            "df": wid.fixed(df),
            "featureName": widFeature,
            "clusterName": wid.fixed(clusterName),
            "clusteringModelName": wid.fixed(clusteringModelName),
            "palette": wid.fixed(palette),
        },
    )

    display(widFeature, out)


def myElbowVisualizer(model, kRange, df, modelName, locate_elbow=False):
    """
    draw 3 elbow charts using YellowBrick KElbowVisualizer with its 3 evaluation scoring metrics Distortion, Silhouette, Calinski-Harabasz

    parameters :
    ------------
    model - scikit-learn model (with attribute n_clusters)
    kRange - list-like : Ks to explore
    df - dataframe : containing features to fit the model on
    modelName - string
    locate_elbow - bool : wether or not to use the locate_elbow from the KElbowVisualizer

    returns :
    ---------
    display a figure with the 3 Elbow charts

    """

    # imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from yellowbrick.cluster import KElbowVisualizer

    # initiate a figure
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs = axs.ravel()

    # create visualizers
    for ax, metric in zip(axs, ["distortion", "silhouette", "calinski_harabasz"]):
        # create a visualizer
        elbowVizRFM = KElbowVisualizer(
            estimator=model,
            k=(kRange[0], kRange[1] + 1),
            metric=metric,
            timings=False,
            locate_elbow=locate_elbow,
            ax=ax,
        )

        # Fit the data to the visualizer
        elbowVizRFM.fit(df)
        elbowVizRFM.finalize()

        # axe Title
        ax.set_title(metric.upper().replace("_", "-") + " Score Elbow")

        # Legend
        if locate_elbow == True:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                borderpad=2,
                frameon=False,
                fontsize=8,
            )

        # label
        ax.set_ylabel("")

        # grid
        ax.grid(axis="y", alpha=0.5, color="grey", linestyle="dotted")
        ax.grid(axis="x", alpha=0)

    # fig title
    plt.suptitle(modelName + "\n")

    return fig


def clusters_plotPairGrid(
    dataframe, palette, sample=None, show_cluster=None, title=None
):
    """
    use Seaborn function "pairgrid"  to plot a 2D pairgrid with hue on clusters

    parameters :
    ------------
    dataframe - dataframe :
                                - first columns : numericals
                                - last column : clusters
    palette - dict : clusters names as keys, colors as values
    sample - int : number of samples used to plot (for better readability). By default : None (no sampling)
    show_cluster - int or list or tuple of clusters names : which clusters to plot
    title - str : pairgrid title. By default : None (no title)


    output :
    --------
    displays a 2D pairgrid, with hue on clusters

    """
    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import gc

    # select clusters labels to plot
    if show_cluster:
        if type(show_cluster) == int:
            show_cluster = [show_cluster]
        if type(show_cluster) == tuple:
            show_cluster = list(show_cluster)
        mask = dataframe.iloc[:, -1].isin(show_cluster)
        filteredDf = dataframe.loc[mask]
    else:
        filteredDf = dataframe.copy()

    # handle sampling
    if sample and sample < len(dataframe):
        df = filteredDf.sample(sample)
    else:
        df = filteredDf.copy()

    del filteredDf
    gc.collect()

    # create seaborn pairgrid with hue on "clusters" column
    g = sns.PairGrid(
        data=df,
        vars=df.columns.to_list()[:-1],
        hue=df.columns.to_list()[-1],
        corner=True,
        palette=palette,
        diag_sharey=False,
    )

    # add 2D scatterplots
    g.map_offdiag(sns.scatterplot, s=10, alpha=0.7)

    # add kdeplots on the diagonal
    g.map_diag(sns.histplot, element="step", fill=False)

    # add legend and move it below the pairgrid
    g.add_legend()
    legend_style = dict(ncols=6, borderpad=5, frameon=False, fontsize=10)
    sns.move_legend(g, "center", bbox_to_anchor=(0.5, 0.9), **legend_style)

    # title
    if not title:
        title = "Clustering - pairplot"
    # if not all clusters plotted, specify
    if show_cluster:
        show_cluster = [str(clusterName) for clusterName in show_cluster]
        if len(show_cluster) == 1:
            title = title + "\n cluster plotted : " + " - ".join(show_cluster)
        else:
            title = title + "\n clusters plotted : " + " - ".join(show_cluster)
    # add title
    g.figure.suptitle(title)

    del df
    gc.collect()

    plt.show()


def clusters_plotPairGrid_WID(dataframe, palette, title=None):
    """
    use "clusters_plotPairGridplot" function to plot a 2D pairgrid with hue on clusters, with options for :
                    - the numbers of samples used to plot
                    - the clusters names to plot

    parameters :
    ------------
    dataframe - dataframe :
                                - first columns : numericals
                                - last column : clusters
    palette - dict : clusters names as keys, colors as values
    title - str : pairgrid title. By default : None (no title)


    output :
    --------
    displays a 2D pairgrid, with hue on clusters

    """

    # imports
    import ipywidgets as wid

    # create a widget for sampling (for plotting purpose)
    widSample = wid.BoundedIntText(
        min=1,
        max=len(dataframe),
        value=5000,
        description="nb of samples : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget to choose which clusters will be plotted

    options = dataframe.iloc[:, -1].cat.categories.to_list()

    widShow_cluster = wid.SelectMultiple(
        options=options,
        rows=len(options),
        description="for plotting, which cluster(s) : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create ui
    ui = wid.HBox(
        [widShow_cluster, widSample], layout=wid.Layout(justify_content="space-between")
    )

    # use interactive_output
    out = wid.interactive_output(
        clusters_plotPairGrid,
        {
            "dataframe": wid.fixed(dataframe),
            "palette": wid.fixed(palette),
            "sample": widSample,
            "show_cluster": widShow_cluster,
            "title": wid.fixed(title),
        },
    )

    display(ui, out)


def plot_scaling_and_Kmeans_pairGrid(
    dataframe,
    n_clusters,
    palette,
    scaler=None,
    random_state=None,
    whichDf="initial",
    title=None,
):
    """
    combine "scaling_and_Kmeans" function and "clusters_plotPairGrid_WID" function to :
                - perform a kmeans clustering on a dataframe
                - plot a 2D pairgrid with hue on clusters

    parameters :
    ------------
    dataframe - dataframe : only numericals
    n_clusters - int
    palette - dict : clusters names as keys, colors as values
    scaler - string : among "std", "rob" and None. By default, None (no scaling)
    random_state - int : random_state parameter for KMeans algorithm. By default, None
    whichDf - string : among "initial" and "scaled" (only if scaler is not None). hence, by default : "initial"
    title - str : pairgrid title. By default : None (no title)

    output :
    --------
    displays a 2D pairgrid, with hue on clusters

    """
    # imports
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # handle scaler parameters and use "scaling_and_Kmeans" function
    if scaler:
        df, df_scaled = scaling_and_Kmeans(
            dataframe=dataframe,
            n_clusters=n_clusters,
            scaler=scaler,
            random_state=random_state,
        )
    else:
        df = scaling_and_Kmeans(
            dataframe=dataframe, n_clusters=n_clusters, random_state=random_state
        )

    # select the dataframe used for plotting
    if whichDf == "initial":
        plotDf = df.copy()
    elif whichDf == "scaled":
        plotDf = df_scaled.copy()

    # "clusters_plotPairGrid_WID" function
    clusters_plotPairGrid_WID(dataframe=plotDf, palette=palette, title=title)


def plot_scaling_and_Kmeans_pairGrid_WID(dataframe, palette, title=None):
    """
    use "plot_scaling_and_Kmeans_pairGrid" to :
                - perform a kmeans clustering on a dataframe
                - plot a 2D pairgrid with hue on clusters
            with options to select :
                - the number of clusters
                - the scaler
                - if scaled, on which dataframe to plot
                - the random_state

    parameters :
    ------------
    dataframe - dataframe : only numericals
    palette - dict : clusters names as keys, colors as values
    title - str : pairgrid title. By default : None (no title)

    output :
    --------
    displays a 2D pairgrid, with hue on clusters

    """

    # imports
    import ipywidgets as wid

    # create a widget for scaling
    widScaler = wid.RadioButtons(
        options={"No scaling": None, "Standard scaler": "std", "Robust scaler": "rob"},
        value="std",
        description="Scaler : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for the number of clusters
    widClusterNumber = wid.BoundedIntText(
        min=1,
        value=1,
        description="nb of cluster : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget for random_state
    widRS = wid.Dropdown(
        options=[None] + [i + 1 for i in range(10)],
        value=None,
        description="Random_state : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # create a widget to choose wether to use original df or scaled one for plotting
    widDf = wid.RadioButtons(
        options=["initial", "scaled"],
        value="initial",
        description="for plotting, which df : ",
        style={"description_width": "initial"},
        disabled=False,
    )

    # handle widScaler changes
    def handleScalerChange(change):
        if change.new == None:
            widDf.options = ["initial"]
        if change.new != None:
            widDf.options = ["initial", "scaled"]
            widDf.value = "initial"

    widScaler.observe(handleScalerChange, "value")

    # create ui
    ui1 = wid.VBox([widClusterNumber, widRS])
    ui = wid.HBox(
        [ui1, widScaler, widDf], layout=wid.Layout(justify_content="space-between")
    )

    # use interactive_output
    out = wid.interactive_output(
        plot_scaling_and_Kmeans_pairGrid,
        {
            "dataframe": wid.fixed(dataframe),
            "n_clusters": widClusterNumber,
            "palette": wid.fixed(palette),
            "scaler": widScaler,
            "random_state": widRS,
            "whichDf": widDf,
            "title": wid.fixed(title),
        },
    )

    display(ui, out)
