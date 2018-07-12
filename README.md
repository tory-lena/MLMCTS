# Multilevel Monte Carlo Tree Search

"Buy low, sell high" is a famous investing adage about taking advantage of the market's propensity to overshoot on the downside and upside. The volatile behavior of stock market prices is an ideal example for an MDP with rare events thht correlate with exceptionally high rewards or penalties. In the area of computational finance, these undulations are often modeled using stochastic processes. Monte Carlo methods are very general and useful approaches for estimating expectations arising from stochastic simulations. However, we not only intend to model them, but to leverage these fluctuations and solve for profitable investment strategies. We will introduce two modifications to the MCTS. To improve its performance on the task, we will use a Multilevel Monte Carlo approach for sampling and, further, append the simulation step of standard MCTS by the splitting method.

If you want to know about the method and how it works, check out the .pdf file. It the methodb build up in the Multilevel Monte Carlo, a research topic in Stats. A julia-implementation of the MLMC method and some application examples for solving SPDEs can be found in the [MLMC repo] of mine, or go directly to the source and check out this [website] by Prof. Mike Giles.

### How to use this repo?

You can download the data folder, just play arround with it (ideas in the stock-model.ipynb) - get familiar with it -, and maybe move on to simulating investment strategies in the Basket-test notebook. Some hyperparamteers are investment duration, starting budget, start-time, look-back window width.

Just have some fun.

[website]: https://people.maths.ox.ac.uk/gilesm/

[MLMC repo]: https://github.com/tory-lena/MLMC_method
