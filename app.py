from dash import Dash, Input, Output, State, callback, dcc, html

from src.utils import (
    load_dummy_columns,
    load_model,
    load_target_enc,
    model_predict,
    transform_data,
)

app = Dash(__name__)

trained_model = load_model("trained_model.pkl")
dummy_columns = load_dummy_columns("dummy_columns.pkl")
y_enc = load_target_enc("y_enc.pkl")
DICT_LABELS = {
    "unacc": "unacceptable",
    "acc": "acceptable",
    "good": "good",
    "vgood": "very good",
}

app.layout = html.Div(
    [
        html.H1(
            children="\U0001F697 Car Evaluation \U0001F697",
            style={"text-align": "center"},
        ),
        html.Div(
            children=[
                "Trained model with the dataset provided by: ",
                html.A(
                    children="https://archive.ics.uci.edu/dataset/19/car+evaluation",
                    href="https://archive.ics.uci.edu/dataset/19/car+evaluation",
                    target="_blank",
                ),
            ],
            style={"text-align": "center"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        "Buying",
                        dcc.Dropdown(
                            options=["vhigh", "high", "med", "low"],
                            value="vhigh",
                            id="buying-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        "Maint",
                        dcc.Dropdown(
                            options=["vhigh", "high", "med", "low"],
                            value="vhigh",
                            id="maint-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        "Doors",
                        dcc.Dropdown(
                            options=["2", "3", "4", "5more"],
                            value="5more",
                            id="doors-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        "Person",
                        dcc.Dropdown(
                            options=["2", "4", "more"],
                            value="2",
                            id="person-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        "Lug Boot",
                        dcc.Dropdown(
                            options=["small", "med", "big"],
                            value="med",
                            id="lugboot-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        "Safety",
                        dcc.Dropdown(
                            options=["low", "med", "high"],
                            value="med",
                            id="safety-dropdown",
                            searchable=False,
                            multi=False,
                        ),
                    ],
                    style={
                        "margin": "auto",
                        "width": "15%",
                        "padding": "10px",
                        "display": "inline-block",
                    },
                ),
            ],
        ),
        html.Div(
            html.Button(
                "Predict", id="predict-button", n_clicks=0, style={"font-size": 20}
            ),
            style={"text-align": "center"},
        ),
        html.Div(
            html.Label("Classification", id="classification-out"),
            style={"text-align": "center", "padding": "20px"},
        ),
    ]
)


@callback(
    Output("classification-out", "children"),
    Input("predict-button", "n_clicks"),
    State("buying-dropdown", "value"),
    State("maint-dropdown", "value"),
    State("doors-dropdown", "value"),
    State("person-dropdown", "value"),
    State("lugboot-dropdown", "value"),
    State("safety-dropdown", "value"),
)
def update_output(n_clicks, buying, maint, doors, person, lugboot, safety):
    data = transform_data(
        buying=buying,
        maint=maint,
        doors=doors,
        person=person,
        lug_boot=lugboot,
        safety=safety,
        dummy_columns=dummy_columns,
    )
    ans = DICT_LABELS[model_predict(model=trained_model, data=data, y_enc=y_enc)]

    return "Classification: {}".format(ans)


if __name__ == "__main__":
    # app.run_server(debug=False, host="0.0.0.0", port=8080)
    app.run(debug=True)
