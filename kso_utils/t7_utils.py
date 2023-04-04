# base imports
import logging

# widget imports
from IPython.display import display
import ipywidgets as widgets

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def choose_text(name: str):
    text_widget = widgets.Text(
        description=f"Please enter a suitable {name} ",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(text_widget)
    return text_widget


class WidgetMaker(widgets.VBox):
    def __init__(self):
        """
        The function creates a widget that allows the user to select which workflows to run

        :param workflows_df: the dataframe of workflows
        """
        self.widget_count = widgets.IntText(
            description="Number of authors:",
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        self.bool_widget_holder = widgets.HBox(
            layout=widgets.Layout(
                width="100%", display="inline-flex", flex_flow="row wrap"
            )
        )
        children = [
            self.widget_count,
            self.bool_widget_holder,
        ]
        self.widget_count.observe(self._add_bool_widgets, names=["value"])
        super().__init__(children=children)

    def _add_bool_widgets(self, widg):
        num_bools = widg["new"]
        new_widgets = []
        for _ in range(num_bools):
            new_widget = widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            ), widgets.Text(
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            )
            new_widget[0].description = "Author Name: " + f" #{_}"
            new_widget[1].description = "Organisation: " + f" #{_}"
            new_widgets.extend(new_widget)
        self.bool_widget_holder.children = tuple(new_widgets)

    @property
    def checks(self):
        return {w.description: w.value for w in self.bool_widget_holder.children}

    @property
    def author_dict(self):
        init_dict = {w.description: w.value for w in self.bool_widget_holder.children}
        names, organisations = [], []
        for i in range(0, len(init_dict), 2):
            names.append(list(init_dict.values())[i])
            organisations.append(list(init_dict.values())[i + 1])
        return {n: org for n, org in zip(names, organisations)}
