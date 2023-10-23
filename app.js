importScripts("https://cdn.jsdelivr.net/pyodide/v0.23.4/pyc/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/1.2.3/dist/wheels/bokeh-3.2.2-py3-none-any.whl', 'https://cdn.holoviz.org/panel/1.2.3/dist/wheels/panel-1.2.3-py3-none-any.whl', 'pyodide-http==0.2.1', 'numpy']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import numpy as np
import panel as pn

from bokeh.plotting import figure, show
from bokeh.models import Legend
from bokeh.layouts import column


# 2. EM Algorithm
class EM:
    def __init__(self, X, n_components=2):
        self.X = X
        self.n = len(X)
        self.n_components = n_components
        self.means = np.linspace(X.min(), X.max(), n_components)
        self.variances = [np.var(X)] * n_components
        self.weights = [1./n_components] * n_components
        self.responsibilities = np.zeros((self.n, n_components))
        self.means_history = []
        self.std_history = []

    def e_step(self):
        for i in range(self.n):
            for k in range(self.n_components):
                self.responsibilities[i, k] = self.weights[k] * self.gaussian_pdf(self.X[i], self.means[k], np.sqrt(self.variances[k]))
            self.responsibilities[i, :] /= sum(self.responsibilities[i, :])

    def m_step(self):
        for k in range(self.n_components):
            N_k = np.sum(self.responsibilities[:, k])
            self.means[k] = np.sum(self.responsibilities[:, k] * self.X) / N_k
            self.variances[k] = np.sum(self.responsibilities[:, k] * (self.X - self.means[k])**2) / N_k
            self.weights[k] = N_k / self.n

    def gaussian_pdf(self, x, mean, std):
        return (1. / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def iteration(self, plot=False):
        self.e_step()
        self.m_step()
        self.means_history.append(self.means.copy())
        self.std_history.append(np.sqrt(self.variances).copy())
        if plot:
            return self.plot_distribution()
            
    def run(self, iterations=50):
        for it in range(iterations):
            self.iteration()
            
    def bic(self):
        """Calculate the Bayesian Information Criterion (BIC)."""
        # Calculate likelihood
        likelihood = np.sum([np.log(np.sum([self.weights[j] * self.gaussian_pdf(self.X[i], self.means[j], np.sqrt(self.variances[j])) for j in range(self.n_components)])) for i in range(self.n)])

        # Number of free parameters: 2 means, 2 variances, and 1 weight (since weights sum to 1)
        k = 3 * self.n_components - 1
        
        # BIC formula
        bic_value = -2 * likelihood + k * np.log(self.n)
        return bic_value
        
    def plot_distribution(self):
        # Create a new figure
        cols = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
        title = f'''
        Data (hist) and Estimated Distributions. 
          Model BIC: {self.bic():2.2f}
          Weights: {np.round(self.weights,2)}
          Means: {np.round(self.means,2)}
          STD: {np.round(np.sqrt(self.variances), 2)}'''

        p = figure(title=title,
                   x_axis_label='Height', y_axis_label='Density', width=800, height=400)

        # Histogram of data
        hist, edges = np.histogram(self.X, density=True, bins=30)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color=None, fill_color='gray', alpha=0.6, legend_label='Data')

        # Estimated distributions
        x = np.linspace(self.X.min(), self.X.max(), 1000)
        legend_items = []
        for k in range(self.n_components):
            y = self.weights[k] * self.gaussian_pdf(x, self.means[k], np.sqrt(self.variances[k]))
            line = p.line(x, y, line_width=2, legend_label=f'Component {k+1}', color=cols[k])  # assuming at most 2 components
            legend_items.append(("Component " + str(k+1), [line]))

        # Setting the legend outside of the figure
        legend = Legend(items=legend_items, location="center")
        p.add_layout(legend, 'right')
        p.legend.click_policy = "hide"

        return p



# Plot the initial distribution

# This will update the plot with each iteration when the button is clicked
def update_plot(event):
    iteration_plot = em.iteration(plot=True)
    layout[1][2] = iteration_plot
    
def set_app(event):
    global em
    em = EM(X, n_components=event.new)
    layout[1][2] = em.plot_distribution()
    return layout



select_num_classes_fit = pn.widgets.IntSlider(name="Number classes fit", start=1, end=4, step=1, value=2) 
button = pn.widgets.Button(name="Click button to run iteration", button_type="primary")
select_num_classes_fit.param.watch(set_app, "value")
button.on_click(update_plot)


global em
explanation = pn.pane.Markdown("""
## LCA explanation 

We are trying to cluster people by their height, with a distribution visualized in the given histogram.

Here is how we created the data:
- There are two groups, with 300 people in the first group, and 1200 in the second.
- Their means are 160 and 200 cm, with a standard deviation of 20 cm.

We want to see if we can recover the groups. First select the number of classes we want to find (you can choose this in the slider).
The algorithm initializes means, weights and variances.

Click on run iteration. 

The algorithm then uses the model to assign each individual to its predicted class (probabilistic).
Then it updates mean, weights and variances.

Keep clicking on run iteration.
""", width=300)


# 1. Generate sample data
np.random.seed(42)
n1, n2 = 300, 1200
mu1, mu2 = 140, 200
sigma = 20

x1 = np.random.normal(mu1, sigma, n1)
x2 = np.random.normal(mu2, sigma, n2)
X = np.concatenate([x1, x2])



em = EM(X, n_components=2)
initial_plot = em.plot_distribution()
layout = pn.Row(explanation, pn.Column(select_num_classes_fit, button, initial_plot) )   

#pn.extension()

layout.servable()



await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.globals.set('patch', msg.patch)
    self.pyodide.runPythonAsync(`
    state.curdoc.apply_json_patch(patch.to_py(), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.globals.set('location', msg.location)
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads(location)
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()