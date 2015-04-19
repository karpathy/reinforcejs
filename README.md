# REINFORCEjs

**REINFORCEjs** is a Reinforcement Learning library that implements several common RL algorithms, all with web demos. In particular, the library currently includes:

- **Dynamic Programming** methods
- (Tabular) **Temporal Difference Learning** (SARSA/Q-Learning)
- **Deep Q-Learning** for Q-Learning with function approximation with Neural Networks
- **Stochastic/Deterministic Policy Gradients** and Actor Critic architectures for dealing with continuous action spaces. (*very alpha, likely buggy or at the very least finicky and inconsistent*)

See the [main webpage](http://cs.stanford.edu/people/karpathy/reinforcejs) for many more details, documentation and demos.

# Code Sketch

The library exports two global variables: `R`, and `RL`. The former contains various kinds of utilities for building expression graphs (e.g. LSTMs) and performing automatic backpropagation, and is a fork of my other project [recurrentjs](https://github.com/karpathy/recurrentjs). The `RL` object contains the current implementations:

- `RL.DPAgent` for finite state/action spaces with environment dynamics
- `RL.TDAgent` for finite state/action spaces
- `RL.DQNAgent` for continuous state features but discrete actions

A typical usage might look something like:

```javascript
// create an environment object
var env = {};
env.getNumStates = function() { return 8; }
env.getMaxNumActions = function() { return 4; }

// create the DQN agent
var spec = { alpha: 0.01 } // see full options on DQN page
agent = new RL.DQNAgent(env, spec); 

setInterval(function(){ // start the learning loop
  var action = agent.act(s); // s is an array of length 8
  //... execute action in environment and get the reward
  agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
}, 0);
```

The full documentation and demos are on the [main webpage](http://cs.stanford.edu/people/karpathy/reinforcejs).

# License

MIT.
