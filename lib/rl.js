var R = {}; // the Recurrent library

(function(global) {
  "use strict";

  // Utility fun
  function assert(condition, message) {
    // from http://stackoverflow.com/questions/15313418/javascript-assert
    if (!condition) {
      message = message || "Assertion failed";
      if (typeof Error !== "undefined") {
        throw new Error(message);
      }
      throw message; // Fallback
    }
  }

  // Random numbers utils
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this
    return_v = true;
    return u*c;
  }
  var randf = function(a, b) { return Math.random()*(b-a)+a; }
  var randi = function(a, b) { return Math.floor(Math.random()*(b-a)+a); }
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  // helper function returns array of zeros of length n
  // and uses typed arrays if available
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = 0; }
      return arr;
    } else {
      return new Float64Array(n);
    }
  }

  // Mat holds a matrix
  var Mat = function(n,d) {
    // n is number of rows d is number of columns
    this.n = n;
    this.d = d;
    this.w = zeros(n * d);
    this.dw = zeros(n * d);
  }
  Mat.prototype = {
    get: function(row, col) { 
      // slow but careful accessor function
      // we want row-major order
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      return this.w[ix];
    },
    set: function(row, col, v) {
      // slow but careful accessor function
      var ix = (this.d * row) + col;
      assert(ix >= 0 && ix < this.w.length);
      this.w[ix] = v; 
    },
    setFrom: function(arr) {
      for(var i=0,n=arr.length;i<n;i++) {
        this.w[i] = arr[i]; 
      }
    },
    setColumn: function(m, i) {
      for(var q=0,n=m.w.length;q<n;q++) {
        this.w[(this.d * q) + i] = m.w[q];
      }
    },
    toJSON: function() {
      var json = {};
      json['n'] = this.n;
      json['d'] = this.d;
      json['w'] = this.w;
      return json;
    },
    fromJSON: function(json) {
      this.n = json.n;
      this.d = json.d;
      this.w = zeros(this.n * this.d);
      this.dw = zeros(this.n * this.d);
      for(var i=0,n=this.n * this.d;i<n;i++) {
        this.w[i] = json.w[i]; // copy over weights
      }
    }
  }

  var copyMat = function(b) {
    var a = new Mat(b.n, b.d);
    a.setFrom(b.w);
    return a;
  }

  var copyNet = function(net) {
    // nets are (k,v) pairs with k = string key, v = Mat()
    var new_net = {};
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        new_net[p] = copyMat(net[p]);
      }
    }
    return new_net;
  }

  var updateMat = function(m, alpha) {
    // updates in place
    for(var i=0,n=m.n*m.d;i<n;i++) {
      if(m.dw[i] !== 0) {
        m.w[i] += - alpha * m.dw[i];
        m.dw[i] = 0;
      }
    }
  }

  var updateNet = function(net, alpha) {
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        updateMat(net[p], alpha);
      }
    }
  }

  var netToJSON = function(net) {
    var j = {};
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        j[p] = net[p].toJSON();
      }
    }
    return j;
  }
  var netFromJSON = function(j) {
    var net = {};
    for(var p in j) {
      if(j.hasOwnProperty(p)){
        net[p] = new Mat(1,1); // not proud of this
        net[p].fromJSON(j[p]);
      }
    }
    return net;
  }
  var netZeroGrads = function(net) {
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        var mat = net[p];
        gradFillConst(mat, 0);
      }
    }
  }
  var netFlattenGrads = function(net) {
    var n = 0;
    for(var p in net) { if(net.hasOwnProperty(p)){ var mat = net[p]; n += mat.dw.length; } }
    var g = new Mat(n, 1);
    var ix = 0;
    for(var p in net) {
      if(net.hasOwnProperty(p)){
        var mat = net[p];
        for(var i=0,m=mat.dw.length;i<m;i++) {
          g.w[ix] = mat.dw[i];
          ix++;
        }
      }
    }
    return g;
  }

  // return Mat but filled with random numbers from gaussian
  var RandMat = function(n,d,mu,std) {
    var m = new Mat(n, d);
    fillRandn(m,mu,std);
    //fillRand(m,-std,std); // kind of :P
    return m;
  }

  // Mat utils
  // fill matrix with random gaussian numbers
  var fillRandn = function(m, mu, std) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randn(mu, std); } }
  var fillRand = function(m, lo, hi) { for(var i=0,n=m.w.length;i<n;i++) { m.w[i] = randf(lo, hi); } }
  var gradFillConst = function(m, c) { for(var i=0,n=m.dw.length;i<n;i++) { m.dw[i] = c } }

  // Transformer definitions
  var Graph = function(needs_backprop) {
    if(typeof needs_backprop === 'undefined') { needs_backprop = true; }
    this.needs_backprop = needs_backprop;

    // this will store a list of functions that perform backprop,
    // in their forward pass order. So in backprop we will go
    // backwards and evoke each one
    this.backprop = [];
  }
  Graph.prototype = {
    backward: function() {
      for(var i=this.backprop.length-1;i>=0;i--) {
        this.backprop[i](); // tick!
      }
    },
    rowPluck: function(m, ix) {
      // pluck a row of m with index ix and return it as col vector
      assert(ix >= 0 && ix < m.n);
      var d = m.d;
      var out = new Mat(d, 1);
      for(var i=0,n=d;i<n;i++){ out.w[i] = m.w[d * ix + i]; } // copy over the data

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=d;i<n;i++){ m.dw[d * ix + i] += out.dw[i]; }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    tanh: function(m) {
      // tanh nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.tanh(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += (1.0 - mwi * mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    sigmoid: function(m) {
      // sigmoid nonlinearity
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = sig(m.w[i]);
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            // grad for z = tanh(x) is (1 - z^2)
            var mwi = out.w[i];
            m.dw[i] += mwi * (1.0 - mwi) * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    relu: function(m) {
      var out = new Mat(m.n, m.d);
      var n = m.w.length;
      for(var i=0;i<n;i++) { 
        out.w[i] = Math.max(0, m.w[i]); // relu
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<n;i++) {
            m.dw[i] += m.w[i] > 0 ? out.dw[i] : 0.0;
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    mul: function(m1, m2) {
      // multiply matrices m1 * m2
      assert(m1.d === m2.n, 'matmul dimensions misaligned');

      var n = m1.n;
      var d = m2.d;
      var out = new Mat(n,d);
      for(var i=0;i<m1.n;i++) { // loop over rows of m1
        for(var j=0;j<m2.d;j++) { // loop over cols of m2
          var dot = 0.0;
          for(var k=0;k<m1.d;k++) { // dot product loop
            dot += m1.w[m1.d*i+k] * m2.w[m2.d*k+j];
          }
          out.w[d*i+j] = dot;
        }
      }

      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0;i<m1.n;i++) { // loop over rows of m1
            for(var j=0;j<m2.d;j++) { // loop over cols of m2
              for(var k=0;k<m1.d;k++) { // dot product loop
                var b = out.dw[d*i+j];
                m1.dw[m1.d*i+k] += m2.w[m2.d*k+j] * b;
                m2.dw[m2.d*k+j] += m1.w[m1.d*i+k] * b;
              }
            }
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    add: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] + m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += out.dw[i];
            m2.dw[i] += out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    dot: function(m1, m2) {
      // m1 m2 are both column vectors
      assert(m1.w.length === m2.w.length);
      var out = new Mat(1,1);
      var dot = 0.0;
      for(var i=0,n=m1.w.length;i<n;i++) {
        dot += m1.w[i] * m2.w[i];
      }
      out.w[0] = dot;
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[0];
            m2.dw[i] += m1.w[i] * out.dw[0];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
    eltmul: function(m1, m2) {
      assert(m1.w.length === m2.w.length);

      var out = new Mat(m1.n, m1.d);
      for(var i=0,n=m1.w.length;i<n;i++) {
        out.w[i] = m1.w[i] * m2.w[i];
      }
      if(this.needs_backprop) {
        var backward = function() {
          for(var i=0,n=m1.w.length;i<n;i++) {
            m1.dw[i] += m2.w[i] * out.dw[i];
            m2.dw[i] += m1.w[i] * out.dw[i];
          }
        }
        this.backprop.push(backward);
      }
      return out;
    },
  }

  var softmax = function(m) {
      var out = new Mat(m.n, m.d); // probability volume
      var maxval = -999999;
      for(var i=0,n=m.w.length;i<n;i++) { if(m.w[i] > maxval) maxval = m.w[i]; }

      var s = 0.0;
      for(var i=0,n=m.w.length;i<n;i++) { 
        out.w[i] = Math.exp(m.w[i] - maxval);
        s += out.w[i];
      }
      for(var i=0,n=m.w.length;i<n;i++) { out.w[i] /= s; }

      // no backward pass here needed
      // since we will use the computed probabilities outside
      // to set gradients directly on m
      return out;
    }

  var Solver = function() {
    this.decay_rate = 0.999;
    this.smooth_eps = 1e-8;
    this.step_cache = {};
  }
  Solver.prototype = {
    step: function(model, step_size, regc, clipval) {
      // perform parameter update
      var solver_stats = {};
      var num_clipped = 0;
      var num_tot = 0;
      for(var k in model) {
        if(model.hasOwnProperty(k)) {
          var m = model[k]; // mat ref
          if(!(k in this.step_cache)) { this.step_cache[k] = new Mat(m.n, m.d); }
          var s = this.step_cache[k];
          for(var i=0,n=m.w.length;i<n;i++) {

            // rmsprop adaptive learning rate
            var mdwi = m.dw[i];
            s.w[i] = s.w[i] * this.decay_rate + (1.0 - this.decay_rate) * mdwi * mdwi;

            // gradient clip
            if(mdwi > clipval) {
              mdwi = clipval;
              num_clipped++;
            }
            if(mdwi < -clipval) {
              mdwi = -clipval;
              num_clipped++;
            }
            num_tot++;

            // update (and regularize)
            m.w[i] += - step_size * mdwi / Math.sqrt(s.w[i] + this.smooth_eps) - regc * m.w[i];
            m.dw[i] = 0; // reset gradients for next iteration
          }
        }
      }
      solver_stats['ratio_clipped'] = num_clipped*1.0/num_tot;
      return solver_stats;
    }
  }

  var initLSTM = function(input_size, hidden_sizes, output_size) {
    // hidden size should be a list

    var model = {};
    for(var d=0;d<hidden_sizes.length;d++) { // loop over depths
      var prev_size = d === 0 ? input_size : hidden_sizes[d - 1];
      var hidden_size = hidden_sizes[d];

      // gates parameters
      model['Wix'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wih'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bi'+d] = new Mat(hidden_size, 1);
      model['Wfx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wfh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bf'+d] = new Mat(hidden_size, 1);
      model['Wox'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Woh'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bo'+d] = new Mat(hidden_size, 1);
      // cell write params
      model['Wcx'+d] = new RandMat(hidden_size, prev_size , 0, 0.08);  
      model['Wch'+d] = new RandMat(hidden_size, hidden_size , 0, 0.08);
      model['bc'+d] = new Mat(hidden_size, 1);
    }
    // decoder params
    model['Whd'] = new RandMat(output_size, hidden_size, 0, 0.08);
    model['bd'] = new Mat(output_size, 1);
    return model;
  }

  var forwardLSTM = function(G, model, hidden_sizes, x, prev) {
    // forward prop for a single tick of LSTM
    // G is graph to append ops to
    // model contains LSTM parameters
    // x is 1D column vector with observation
    // prev is a struct containing hidden and cell
    // from previous iteration

    if(prev == null || typeof prev.h === 'undefined') {
      var hidden_prevs = [];
      var cell_prevs = [];
      for(var d=0;d<hidden_sizes.length;d++) {
        hidden_prevs.push(new R.Mat(hidden_sizes[d],1)); 
        cell_prevs.push(new R.Mat(hidden_sizes[d],1)); 
      }
    } else {
      var hidden_prevs = prev.h;
      var cell_prevs = prev.c;
    }

    var hidden = [];
    var cell = [];
    for(var d=0;d<hidden_sizes.length;d++) {

      var input_vector = d === 0 ? x : hidden[d-1];
      var hidden_prev = hidden_prevs[d];
      var cell_prev = cell_prevs[d];

      // input gate
      var h0 = G.mul(model['Wix'+d], input_vector);
      var h1 = G.mul(model['Wih'+d], hidden_prev);
      var input_gate = G.sigmoid(G.add(G.add(h0,h1),model['bi'+d]));

      // forget gate
      var h2 = G.mul(model['Wfx'+d], input_vector);
      var h3 = G.mul(model['Wfh'+d], hidden_prev);
      var forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf'+d]));

      // output gate
      var h4 = G.mul(model['Wox'+d], input_vector);
      var h5 = G.mul(model['Woh'+d], hidden_prev);
      var output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo'+d]));

      // write operation on cells
      var h6 = G.mul(model['Wcx'+d], input_vector);
      var h7 = G.mul(model['Wch'+d], hidden_prev);
      var cell_write = G.tanh(G.add(G.add(h6, h7),model['bc'+d]));

      // compute new cell activation
      var retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
      var write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
      var cell_d = G.add(retain_cell, write_cell); // new cell contents

      // compute hidden state as gated, saturated cell activations
      var hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

      hidden.push(hidden_d);
      cell.push(cell_d);
    }

    // one decoder to outputs at end
    var output = G.add(G.mul(model['Whd'], hidden[hidden.length - 1]),model['bd']);

    // return cell memory, hidden representation and output
    return {'h':hidden, 'c':cell, 'o' : output};
  }

  var sig = function(x) {
    // helper function for computing sigmoid
    return 1.0/(1+Math.exp(-x));
  }

  var maxi = function(w) {
    // argmax of array w
    var maxv = w[0];
    var maxix = 0;
    for(var i=1,n=w.length;i<n;i++) {
      var v = w[i];
      if(v > maxv) {
        maxix = i;
        maxv = v;
      }
    }
    return maxix;
  }

  var samplei = function(w) {
    // sample argmax from w, assuming w are 
    // probabilities that sum to one
    var r = randf(0,1);
    var x = 0.0;
    var i = 0;
    while(true) {
      x += w[i];
      if(x > r) { return i; }
      i++;
    }
    return w.length - 1; // pretty sure we should never get here?
  }

  // various utils
  global.assert = assert;
  global.zeros = zeros;
  global.maxi = maxi;
  global.samplei = samplei;
  global.randi = randi;
  global.randn = randn;
  global.softmax = softmax;
  // classes
  global.Mat = Mat;
  global.RandMat = RandMat;
  global.forwardLSTM = forwardLSTM;
  global.initLSTM = initLSTM;
  // more utils
  global.updateMat = updateMat;
  global.updateNet = updateNet;
  global.copyMat = copyMat;
  global.copyNet = copyNet;
  global.netToJSON = netToJSON;
  global.netFromJSON = netFromJSON;
  global.netZeroGrads = netZeroGrads;
  global.netFlattenGrads = netFlattenGrads;
  // optimization
  global.Solver = Solver;
  global.Graph = Graph;
})(R);

// END OF RECURRENTJS

var RL = {};
(function(global) {
  "use strict";

// syntactic sugar function for getting default parameter values
var getopt = function(opt, field_name, default_value) {
  if(typeof opt === 'undefined') { return default_value; }
  return (typeof opt[field_name] !== 'undefined') ? opt[field_name] : default_value;
}

var zeros = R.zeros; // inherit these
var assert = R.assert;
var randi = R.randi;
var randf = R.randf;

var setConst = function(arr, c) {
  for(var i=0,n=arr.length;i<n;i++) {
    arr[i] = c;
  }
}

var sampleWeighted = function(p) {
  var r = Math.random();
  var c = 0.0;
  for(var i=0,n=p.length;i<n;i++) {
    c += p[i];
    if(c >= r) { return i; }
  }
  assert(false, 'wtf');
}

// ------
// AGENTS
// ------

// DPAgent performs Value Iteration
// - can also be used for Policy Iteration if you really wanted to
// - requires model of the environment :(
// - does not learn from experience :(
// - assumes finite MDP :(
var DPAgent = function(env, opt) {
  this.V = null; // state value function
  this.P = null; // policy distribution \pi(s,a)
  this.env = env; // store pointer to environment
  this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
  this.reset();
}
DPAgent.prototype = {
  reset: function() {
    // reset the agent's policy and value function
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.V = zeros(this.ns);
    this.P = zeros(this.ns * this.na);
    // initialize uniform random policy
    for(var s=0;s<this.ns;s++) {
      var poss = this.env.allowedActions(s);
      for(var i=0,n=poss.length;i<n;i++) {
        this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
      }
    }
  },
  act: function(s) {
    // behave according to the learned policy
    var poss = this.env.allowedActions(s);
    var ps = [];
    for(var i=0,n=poss.length;i<n;i++) {
      var a = poss[i];
      var prob = this.P[a*this.ns+s];
      ps.push(prob);
    }
    var maxi = sampleWeighted(ps);
    return poss[maxi];
  },
  learn: function() {
    // perform a single round of value iteration
    self.evaluatePolicy(); // writes this.V
    self.updatePolicy(); // writes this.P
  },
  evaluatePolicy: function() {
    // perform a synchronous update of the value function
    var Vnew = zeros(this.ns);
    for(var s=0;s<this.ns;s++) {
      // integrate over actions in a stochastic policy
      // note that we assume that policy probability mass over allowed actions sums to one
      var v = 0.0;
      var poss = this.env.allowedActions(s);
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var prob = this.P[a*this.ns+s]; // probability of taking action under policy
        if(prob === 0) { continue; } // no contribution, skip for speed
        var ns = this.env.nextStateDistribution(s,a);
        var rs = this.env.reward(s,a,ns); // reward for s->a->ns transition
        v += prob * (rs + this.gamma * this.V[ns]);
      }
      Vnew[s] = v;
    }
    this.V = Vnew; // swap
  },
  updatePolicy: function() {
    // update policy to be greedy w.r.t. learned Value function
    for(var s=0;s<this.ns;s++) {
      var poss = this.env.allowedActions(s);
      // compute value of taking each allowed action
      var vmax, nmax;
      var vs = [];
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        var ns = this.env.nextStateDistribution(s,a);
        var rs = this.env.reward(s,a,ns);
        var v = rs + this.gamma * this.V[ns];
        vs.push(v);
        if(i === 0 || v > vmax) { vmax = v; nmax = 1; }
        else if(v === vmax) { nmax += 1; }
      }
      // update policy smoothly across all argmaxy actions
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        this.P[a*this.ns+s] = (vs[i] === vmax) ? 1.0/nmax : 0.0;
      }
    }
  },
}

// QAgent uses TD (Q-Learning, SARSA)
// - does not require environment model :)
// - learns from experience :)
var TDAgent = function(env, opt) {
  this.update = getopt(opt, 'update', 'qlearn'); // qlearn | sarsa
  this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

  // class allows non-deterministic policy, and smoothly regressing towards the optimal policy based on Q
  this.smooth_policy_update = getopt(opt, 'smooth_policy_update', false);
  this.beta = getopt(opt, 'beta', 0.01); // learning rate for policy, if smooth updates are on

  // eligibility traces
  this.lambda = getopt(opt, 'lambda', 0); // eligibility trace decay. 0 = no eligibility traces used
  this.replacing_traces = getopt(opt, 'replacing_traces', true);

  // optional optimistic initial values
  this.q_init_val = getopt(opt, 'q_init_val', 0);

  this.planN = getopt(opt, 'planN', 0); // number of planning steps per learning iteration (0 = no planning)

  this.Q = null; // state action value function
  this.P = null; // policy distribution \pi(s,a)
  this.e = null; // eligibility trace
  this.env_model_s = null;; // environment model (s,a) -> (s',r)
  this.env_model_r = null;; // environment model (s,a) -> (s',r)
  this.env = env; // store pointer to environment
  this.reset();
}
TDAgent.prototype = {
  reset: function(){
    // reset the agent's policy and value function
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.Q = zeros(this.ns * this.na);
    if(this.q_init_val !== 0) { setConst(this.Q, this.q_init_val); }
    this.P = zeros(this.ns * this.na);
    this.e = zeros(this.ns * this.na);

    // model/planning vars
    this.env_model_s = zeros(this.ns * this.na);
    setConst(this.env_model_s, -1); // init to -1 so we can test if we saw the state before
    this.env_model_r = zeros(this.ns * this.na);
    this.sa_seen = [];
    this.pq = zeros(this.ns * this.na);

    // initialize uniform random policy
    for(var s=0;s<this.ns;s++) {
      var poss = this.env.allowedActions(s);
      for(var i=0,n=poss.length;i<n;i++) {
        this.P[poss[i]*this.ns+s] = 1.0 / poss.length;
      }
    }
    // agent memory, needed for streaming updates
    // (s0,a0,r0,s1,a1,r1,...)
    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  },
  resetEpisode: function() {
    // an episode finished
  },
  act: function(s){
    // act according to epsilon greedy policy
    var poss = this.env.allowedActions(s);
    var probs = [];
    for(var i=0,n=poss.length;i<n;i++) {
      probs.push(this.P[poss[i]*this.ns+s]);
    }
    // epsilon greedy policy
    if(Math.random() < this.epsilon) {
      var a = poss[randi(0,poss.length)]; // random available action
      this.explored = true;
    } else {
      var a = poss[sampleWeighted(probs)];
      this.explored = false;
    }
    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
  },
  learn: function(r1){
    // takes reward for previous action, which came from a call to act()
    if(!(this.r0 == null)) {
      this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1, this.lambda);
      if(this.planN > 0) {
        this.updateModel(this.s0, this.a0, this.r0, this.s1);
        this.plan();
      }
    }
    this.r0 = r1; // store this for next update
  },
  updateModel: function(s0, a0, r0, s1) {
    // transition (s0,a0) -> (r0,s1) was observed. Update environment model
    var sa = a0 * this.ns + s0;
    if(this.env_model_s[sa] === -1) {
      // first time we see this state action
      this.sa_seen.push(a0 * this.ns + s0); // add as seen state
    }
    this.env_model_s[sa] = s1;
    this.env_model_r[sa] = r0;
  },
  plan: function() {

    // order the states based on current priority queue information
    var spq = [];
    for(var i=0,n=this.sa_seen.length;i<n;i++) {
      var sa = this.sa_seen[i];
      var sap = this.pq[sa];
      if(sap > 1e-5) { // gain a bit of efficiency
        spq.push({sa:sa, p:sap});
      }
    }
    spq.sort(function(a,b){ return a.p < b.p ? 1 : -1});

    // perform the updates
    var nsteps = Math.min(this.planN, spq.length);
    for(var k=0;k<nsteps;k++) {
      // random exploration
      //var i = randi(0, this.sa_seen.length); // pick random prev seen state action
      //var s0a0 = this.sa_seen[i];
      var s0a0 = spq[k].sa;
      this.pq[s0a0] = 0; // erase priority, since we're backing up this state
      var s0 = s0a0 % this.ns;
      var a0 = Math.floor(s0a0 / this.ns);
      var r0 = this.env_model_r[s0a0];
      var s1 = this.env_model_s[s0a0];
      var a1 = -1; // not used for Q learning
      if(this.update === 'sarsa') {
        // generate random action?...
        var poss = this.env.allowedActions(s1);
        var a1 = poss[randi(0,poss.length)];
      }
      this.learnFromTuple(s0, a0, r0, s1, a1, 0); // note lambda = 0 - shouldnt use eligibility trace here
    }
  },
  learnFromTuple: function(s0, a0, r0, s1, a1, lambda) {
    var sa = a0 * this.ns + s0;

    // calculate the target for Q(s,a)
    if(this.update === 'qlearn') {
      // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
      var poss = this.env.allowedActions(s1);
      var qmax = 0;
      for(var i=0,n=poss.length;i<n;i++) {
        var s1a = poss[i] * this.ns + s1;
        var qval = this.Q[s1a];
        if(i === 0 || qval > qmax) { qmax = qval; }
      }
      var target = r0 + this.gamma * qmax;
    } else if(this.update === 'sarsa') {
      // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
      var s1a1 = a1 * this.ns + s1;
      var target = r0 + this.gamma * this.Q[s1a1];
    }

    if(lambda > 0) {
      // perform an eligibility trace update
      if(this.replacing_traces) {
        this.e[sa] = 1;
      } else {
        this.e[sa] += 1;
      }
      var edecay = lambda * this.gamma;
      var state_update = zeros(this.ns);
      for(var s=0;s<this.ns;s++) {
        var poss = this.env.allowedActions(s);
        for(var i=0;i<poss.length;i++) {
          var a = poss[i];
          var saloop = a * this.ns + s;
          var esa = this.e[saloop];
          var update = this.alpha * esa * (target - this.Q[saloop]);
          this.Q[saloop] += update;
          this.updatePriority(s, a, update);
          this.e[saloop] *= edecay;
          var u = Math.abs(update);
          if(u > state_update[s]) { state_update[s] = u; }
        }
      }
      for(var s=0;s<this.ns;s++) {
        if(state_update[s] > 1e-5) { // save efficiency here
          this.updatePolicy(s);
        }
      }
      if(this.explored && this.update === 'qlearn') {
        // have to wipe the trace since q learning is off-policy :(
        this.e = zeros(this.ns * this.na);
      }
    } else {
      // simpler and faster update without eligibility trace
      // update Q[sa] towards it with some step size
      var update = this.alpha * (target - this.Q[sa]);
      this.Q[sa] += update;
      this.updatePriority(s0, a0, update);
      // update the policy to reflect the change (if appropriate)
      this.updatePolicy(s0);
    }
  },
  updatePriority: function(s,a,u) {
    // used in planning. Invoked when Q[sa] += update
    // we should find all states that lead to (s,a) and upgrade their priority
    // of being update in the next planning step
    u = Math.abs(u);
    if(u < 1e-5) { return; } // for efficiency skip small updates
    if(this.planN === 0) { return; } // there is no planning to be done, skip.
    for(var si=0;si<this.ns;si++) {
      // note we are also iterating over impossible actions at all states,
      // but this should be okay because their env_model_s should simply be -1
      // as initialized, so they will never be predicted to point to any state
      // because they will never be observed, and hence never be added to the model
      for(var ai=0;ai<this.na;ai++) {
        var siai = ai * this.ns + si;
        if(this.env_model_s[siai] === s) {
          // this state leads to s, add it to priority queue
          this.pq[siai] += u;
        }
      }
    }
  },
  updatePolicy: function(s) {
    var poss = this.env.allowedActions(s);
    // set policy at s to be the action that achieves max_a Q(s,a)
    // first find the maxy Q values
    var qmax, nmax;
    var qs = [];
    for(var i=0,n=poss.length;i<n;i++) {
      var a = poss[i];
      var qval = this.Q[a*this.ns+s];
      qs.push(qval);
      if(i === 0 || qval > qmax) { qmax = qval; nmax = 1; }
      else if(qval === qmax) { nmax += 1; }
    }
    // now update the policy smoothly towards the argmaxy actions
    var psum = 0.0;
    for(var i=0,n=poss.length;i<n;i++) {
      var a = poss[i];
      var target = (qs[i] === qmax) ? 1.0/nmax : 0.0;
      var ix = a*this.ns+s;
      if(this.smooth_policy_update) {
        // slightly hacky :p
        this.P[ix] += this.beta * (target - this.P[ix]);
        psum += this.P[ix];
      } else {
        // set hard target
        this.P[ix] = target;
      }
    }
    if(this.smooth_policy_update) {
      // renomalize P if we're using smooth policy updates
      for(var i=0,n=poss.length;i<n;i++) {
        var a = poss[i];
        this.P[a*this.ns+s] /= psum;
      }
    }
  }
}


var DQNAgent = function(env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.75); // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.01); // value function learning rate

  this.experience_add_every = getopt(opt, 'experience_add_every', 25); // number of time steps before we add another experience to replay memory
  this.experience_size = getopt(opt, 'experience_size', 5000); // size of experience replay
  this.learning_steps_per_iteration = getopt(opt, 'learning_steps_per_iteration', 10);
  this.tderror_clamp = getopt(opt, 'tderror_clamp', 1.0); 

  this.num_hidden_units =  getopt(opt, 'num_hidden_units', 100); 

  this.env = env;
  this.reset();
}
DQNAgent.prototype = {
  reset: function() {
    this.nh = this.num_hidden_units; // number of hidden units
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();

    // nets are hardcoded for now as key (str) -> Mat
    // not proud of this. better solution is to have a whole Net object
    // on top of Mats, but for now sticking with this
    this.net = {};
    this.net.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
    this.net.b1 = new R.Mat(this.nh, 1, 0, 0.01);
    this.net.W2 = new R.RandMat(this.na, this.nh, 0, 0.01);
    this.net.b2 = new R.Mat(this.na, 1, 0, 0.01);

    this.exp = []; // experience
    this.expi = 0; // where to insert

    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;

    this.tderror = 0; // for visualization only...
  },
  toJSON: function() {
    // save function
    var j = {};
    j.nh = this.nh;
    j.ns = this.ns;
    j.na = this.na;
    j.net = R.netToJSON(this.net);
    return j;
  },
  fromJSON: function(j) {
    // load function
    this.nh = j.nh;
    this.ns = j.ns;
    this.na = j.na;
    this.net = R.netFromJSON(j.net);
  },
  forwardQ: function(net, s, needs_backprop) {
    var G = new R.Graph(needs_backprop);
    var a1mat = G.add(G.mul(net.W1, s), net.b1);
    var h1mat = G.tanh(a1mat);
    var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    this.lastG = G; // back this up. Kind of hacky isn't it
    return a2mat;
  },
  act: function(slist) {
    // convert to a Mat column vector
    var s = new R.Mat(this.ns, 1);
    s.setFrom(slist);

    // epsilon greedy policy
    if(Math.random() < this.epsilon) {
      var a = randi(0, this.na);
    } else {
      // greedy wrt Q function
      var amat = this.forwardQ(this.net, s, false);
      var a = R.maxi(amat.w); // returns index of argmax action
    }

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  },
  learn: function(r1) {
    // perform an update on Q function
    if(!(this.r0 == null) && this.alpha > 0) {

      // learn from this tuple to get a sense of how "surprising" it is to the agent
      var tderror = this.learnFromTuple(this.s0, this.a0, this.r0, this.s1, this.a1);
      this.tderror = tderror; // a measure of surprise

      // decide if we should keep this experience in the replay
      if(this.t % this.experience_add_every === 0) {
        this.exp[this.expi] = [this.s0, this.a0, this.r0, this.s1, this.a1];
        this.expi += 1;
        if(this.expi > this.experience_size) { this.expi = 0; } // roll over when we run out
      }
      this.t += 1;

      // sample some additional experience from replay memory and learn from it
      for(var k=0;k<this.learning_steps_per_iteration;k++) {
        var ri = randi(0, this.exp.length); // todo: priority sweeps?
        var e = this.exp[ri];
        this.learnFromTuple(e[0], e[1], e[2], e[3], e[4])
      }
    }
    this.r0 = r1; // store for next update
  },
  learnFromTuple: function(s0, a0, r0, s1, a1) {
    // want: Q(s,a) = r + gamma * max_a' Q(s',a')

    // compute the target Q value
    var tmat = this.forwardQ(this.net, s1, false);
    var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

    // now predict
    var pred = this.forwardQ(this.net, s0, true);

    var tderror = pred.w[a0] - qmax;
    var clamp = this.tderror_clamp;
    if(Math.abs(tderror) > clamp) {  // huber loss to robustify
      if(tderror > clamp) tderror = clamp;
      if(tderror < -clamp) tderror = -clamp;
    }
    pred.dw[a0] = tderror;
    this.lastG.backward(); // compute gradients on net params

    // update net
    R.updateNet(this.net, this.alpha);
    return tderror;
  }
}

// buggy implementation, doesnt work...
var SimpleReinforceAgent = function(env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.75); // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
  this.env = env;
  this.reset();
}
SimpleReinforceAgent.prototype = {
  reset: function() {
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.nh = 100; // number of hidden units
    this.nhb = 100; // and also in the baseline lstm

    this.actorNet = {};
    this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
    this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
    this.actorNet.W2 = new R.RandMat(this.na, this.nh, 0, 0.1);
    this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
    this.actorOutputs = [];
    this.actorGraphs = [];
    this.actorActions = []; // sampled ones

    this.rewardHistory = [];
    
    this.baselineNet = {};
    this.baselineNet.W1 = new R.RandMat(this.nhb, this.ns, 0, 0.01);
    this.baselineNet.b1 = new R.Mat(this.nhb, 1, 0, 0.01);
    this.baselineNet.W2 = new R.RandMat(this.na, this.nhb, 0, 0.01);
    this.baselineNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
    this.baselineOutputs = [];
    this.baselineGraphs = [];

    this.t = 0;
  },
  forwardActor: function(s, needs_backprop) {
    var net = this.actorNet;
    var G = new R.Graph(needs_backprop);
    var a1mat = G.add(G.mul(net.W1, s), net.b1);
    var h1mat = G.tanh(a1mat);
    var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    return {'a':a2mat, 'G':G}
  },
  forwardValue: function(s, needs_backprop) {
    var net = this.baselineNet;
    var G = new R.Graph(needs_backprop);
    var a1mat = G.add(G.mul(net.W1, s), net.b1);
    var h1mat = G.tanh(a1mat);
    var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    return {'a':a2mat, 'G':G}
  },
  act: function(slist) {
    // convert to a Mat column vector
    var s = new R.Mat(this.ns, 1);
    s.setFrom(slist);

    // forward the actor to get action output
    var ans = this.forwardActor(s, true);
    var amat = ans.a;
    var ag = ans.G;
    this.actorOutputs.push(amat);
    this.actorGraphs.push(ag);

    // forward the baseline estimator
    var ans = this.forwardValue(s, true);
    var vmat = ans.a;
    var vg = ans.G;
    this.baselineOutputs.push(vmat);
    this.baselineGraphs.push(vg);

    // sample action from the stochastic gaussian policy
    var a = R.copyMat(amat);
    var gaussVar = 0.02;
    a.w[0] = R.randn(0, gaussVar);
    a.w[1] = R.randn(0, gaussVar);
  
    this.actorActions.push(a);

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  },
  learn: function(r1) {
    // perform an update on Q function
    this.rewardHistory.push(r1);
    var n = this.rewardHistory.length;
    var baselineMSE = 0.0;
    var nup = 100; // what chunk of experience to take
    var nuse = 80; // what chunk to update from
    if(n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      var vs = [];
      for(var t=0;t<nuse;t++) {
        var mul = 1;
        // compute the actual discounted reward for this time step
        var V = 0;
        for(var t2=t;t2<n;t2++) {
          V += mul * this.rewardHistory[t2];
          mul *= this.gamma;
          if(mul < 1e-5) { break; } // efficiency savings
        }
        // get the predicted baseline at this time step
        var b = this.baselineOutputs[t].w[0];
        for(var i=0;i<this.na;i++) {
          // [the action delta] * [the desirebility]
          var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
          if(update > 0.1) { update = 0.1; }
          if(update < -0.1) { update = -0.1; }
          this.actorOutputs[t].dw[i] += update;
        }
        var update = - (V - b);
        if(update > 0.1) { update = 0.1; }
        if(update < 0.1) { update = -0.1; }
        this.baselineOutputs[t].dw[0] += update;
        baselineMSE += (V - b) * (V - b);
        vs.push(V);
      }
      baselineMSE /= nuse;
      // backprop all the things
      for(var t=0;t<nuse;t++) {
        this.actorGraphs[t].backward();
        this.baselineGraphs[t].backward();
      }
      R.updateNet(this.actorNet, this.alpha); // update actor network
      R.updateNet(this.baselineNet, this.beta); // update baseline network

      // flush
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];
      this.baselineOutputs = [];
      this.actorGraphs = [];
      this.baselineGraphs = [];

      this.tderror = baselineMSE;
    }
    this.t += 1;
    this.r0 = r1; // store for next update
  },
}

// buggy implementation as well, doesn't work
var RecurrentReinforceAgent = function(env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.1); // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
  this.env = env;
  this.reset();
}
RecurrentReinforceAgent.prototype = {
  reset: function() {
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.nh = 40; // number of hidden units
    this.nhb = 40; // and also in the baseline lstm

    this.actorLSTM = R.initLSTM(this.ns, [this.nh], this.na);
    this.actorG = new R.Graph();
    this.actorPrev = null;
    this.actorOutputs = [];
    this.rewardHistory = [];
    this.actorActions = [];

    this.baselineLSTM = R.initLSTM(this.ns, [this.nhb], 1);
    this.baselineG = new R.Graph();
    this.baselinePrev = null;
    this.baselineOutputs = [];

    this.t = 0;

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
  },
  act: function(slist) {
    // convert to a Mat column vector
    var s = new R.Mat(this.ns, 1);
    s.setFrom(slist);

    // forward the LSTM to get action distribution
    var actorNext = R.forwardLSTM(this.actorG, this.actorLSTM, [this.nh], s, this.actorPrev);
    this.actorPrev = actorNext;
    var amat = actorNext.o;
    this.actorOutputs.push(amat);

    // forward the baseline LSTM
    var baselineNext = R.forwardLSTM(this.baselineG, this.baselineLSTM, [this.nhb], s, this.baselinePrev);
    this.baselinePrev = baselineNext;
    this.baselineOutputs.push(baselineNext.o);

    // sample action from actor policy
    var gaussVar = 0.05;
    var a = R.copyMat(amat);
    for(var i=0,n=a.w.length;i<n;i++) {
      a.w[0] += R.randn(0, gaussVar);
      a.w[1] += R.randn(0, gaussVar);
    }
    this.actorActions.push(a);

    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;
    return a;
  },
  learn: function(r1) {
    // perform an update on Q function
    this.rewardHistory.push(r1);
    var n = this.rewardHistory.length;
    var baselineMSE = 0.0;
    var nup = 100; // what chunk of experience to take
    var nuse = 80; // what chunk to also update
    if(n >= nup) {
      // lets learn and flush
      // first: compute the sample values at all points
      var vs = [];
      for(var t=0;t<nuse;t++) {
        var mul = 1;
        var V = 0;
        for(var t2=t;t2<n;t2++) {
          V += mul * this.rewardHistory[t2];
          mul *= this.gamma;
          if(mul < 1e-5) { break; } // efficiency savings
        }
        var b = this.baselineOutputs[t].w[0];
        // todo: take out the constants etc.
        for(var i=0;i<this.na;i++) {
          // [the action delta] * [the desirebility]
          var update = - (V - b) * (this.actorActions[t].w[i] - this.actorOutputs[t].w[i]);
          if(update > 0.1) { update = 0.1; }
          if(update < -0.1) { update = -0.1; }
          this.actorOutputs[t].dw[i] += update;
        }
        var update = - (V - b);
        if(update > 0.1) { update = 0.1; }
        if(update < 0.1) { update = -0.1; }
        this.baselineOutputs[t].dw[0] += update;
        baselineMSE += (V-b)*(V-b);
        vs.push(V);
      }
      baselineMSE /= nuse;
      this.actorG.backward(); // update params! woohoo!
      this.baselineG.backward();
      R.updateNet(this.actorLSTM, this.alpha); // update actor network
      R.updateNet(this.baselineLSTM, this.beta); // update baseline network

      // flush
      this.actorG = new R.Graph();
      this.actorPrev = null;
      this.actorOutputs = [];
      this.rewardHistory = [];
      this.actorActions = [];

      this.baselineG = new R.Graph();
      this.baselinePrev = null;
      this.baselineOutputs = [];

      this.tderror = baselineMSE;
    }
    this.t += 1;
    this.r0 = r1; // store for next update
  },
}

// Currently buggy implementation, doesnt work
var DeterministPG = function(env, opt) {
  this.gamma = getopt(opt, 'gamma', 0.5); // future reward discount factor
  this.epsilon = getopt(opt, 'epsilon', 0.5); // for epsilon-greedy policy
  this.alpha = getopt(opt, 'alpha', 0.001); // actor net learning rate
  this.beta = getopt(opt, 'beta', 0.01); // baseline net learning rate
  this.env = env;
  this.reset();
}
DeterministPG.prototype = {
  reset: function() {
    this.ns = this.env.getNumStates();
    this.na = this.env.getMaxNumActions();
    this.nh = 100; // number of hidden units

    // actor
    this.actorNet = {};
    this.actorNet.W1 = new R.RandMat(this.nh, this.ns, 0, 0.01);
    this.actorNet.b1 = new R.Mat(this.nh, 1, 0, 0.01);
    this.actorNet.W2 = new R.RandMat(this.na, this.ns, 0, 0.1);
    this.actorNet.b2 = new R.Mat(this.na, 1, 0, 0.01);
    this.ntheta = this.na*this.ns+this.na; // number of params in actor

    // critic
    this.criticw = new R.RandMat(1, this.ntheta, 0, 0.01); // row vector

    this.r0 = null;
    this.s0 = null;
    this.s1 = null;
    this.a0 = null;
    this.a1 = null;
    this.t = 0;
  },
  forwardActor: function(s, needs_backprop) {
    var net = this.actorNet;
    var G = new R.Graph(needs_backprop);
    var a1mat = G.add(G.mul(net.W1, s), net.b1);
    var h1mat = G.tanh(a1mat);
    var a2mat = G.add(G.mul(net.W2, h1mat), net.b2);
    return {'a':a2mat, 'G':G}
  },
  act: function(slist) {
    // convert to a Mat column vector
    var s = new R.Mat(this.ns, 1);
    s.setFrom(slist);

    // forward the actor to get action output
    var ans = this.forwardActor(s, false);
    var amat = ans.a;
    var ag = ans.G;

    // sample action from the stochastic gaussian policy
    var a = R.copyMat(amat);
    if(Math.random() < this.epsilon) {
      var gaussVar = 0.02;
      a.w[0] = R.randn(0, gaussVar);
      a.w[1] = R.randn(0, gaussVar);
    }
    var clamp = 0.25;
    if(a.w[0] > clamp) a.w[0] = clamp;
    if(a.w[0] < -clamp) a.w[0] = -clamp;
    if(a.w[1] > clamp) a.w[1] = clamp;
    if(a.w[1] < -clamp) a.w[1] = -clamp;
  
    // shift state memory
    this.s0 = this.s1;
    this.a0 = this.a1;
    this.s1 = s;
    this.a1 = a;

    return a;
  },
  utilJacobianAt: function(s) {
    var ujacobian = new R.Mat(this.ntheta, this.na);
    for(var a=0;a<this.na;a++) {
      R.netZeroGrads(this.actorNet);
      var ag = this.forwardActor(this.s0, true);
      ag.a.dw[a] = 1.0;
      ag.G.backward();
      var gflat = R.netFlattenGrads(this.actorNet);
      ujacobian.setColumn(gflat,a);
    }
    return ujacobian;
  },
  learn: function(r1) {
    // perform an update on Q function
    //this.rewardHistory.push(r1);
    if(!(this.r0 == null)) {
      var Gtmp = new R.Graph(false);
      // dpg update:
      // first compute the features psi:
      // the jacobian matrix of the actor for s
      var ujacobian0 = this.utilJacobianAt(this.s0);
      // now form the features \psi(s,a)
      var psi_sa0 = Gtmp.mul(ujacobian0, this.a0); // should be [this.ntheta x 1] "feature" vector
      var qw0 = Gtmp.mul(this.criticw, psi_sa0); // 1x1
      // now do the same thing because we need \psi(s_{t+1}, \mu\_\theta(s\_t{t+1}))
      var ujacobian1 = this.utilJacobianAt(this.s1);
      var ag = this.forwardActor(this.s1, false);
      var psi_sa1 = Gtmp.mul(ujacobian1, ag.a);
      var qw1 = Gtmp.mul(this.criticw, psi_sa1); // 1x1
      // get the td error finally
      var tderror = this.r0 + this.gamma * qw1.w[0] - qw0.w[0]; // lol
      if(tderror > 0.5) tderror = 0.5; // clamp
      if(tderror < -0.5) tderror = -0.5;
      this.tderror = tderror;

      // update actor policy with natural gradient
      var net = this.actorNet;
      var ix = 0;
      for(var p in net) {
        var mat = net[p];
        if(net.hasOwnProperty(p)){
          for(var i=0,n=mat.w.length;i<n;i++) {
            mat.w[i] += this.alpha * this.criticw.w[ix]; // natural gradient update
            ix+=1;
          }
        }
      }
      // update the critic parameters too
      for(var i=0;i<this.ntheta;i++) {
        var update = this.beta * tderror * psi_sa0.w[i];
        this.criticw.w[i] += update;
      }
    }
    this.r0 = r1; // store for next update
  },
}

// exports
global.DPAgent = DPAgent;
global.TDAgent = TDAgent;
global.DQNAgent = DQNAgent;
//global.SimpleReinforceAgent = SimpleReinforceAgent;
//global.RecurrentReinforceAgent = RecurrentReinforceAgent;
//global.DeterministPG = DeterministPG;
})(RL);


