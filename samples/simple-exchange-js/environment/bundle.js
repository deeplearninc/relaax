/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};

/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {

/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId])
/******/ 			return installedModules[moduleId].exports;

/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};

/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);

/******/ 		// Flag the module as loaded
/******/ 		module.l = true;

/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}


/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;

/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;

/******/ 	// identity function for calling harmony imports with the correct context
/******/ 	__webpack_require__.i = function(value) { return value; };

/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};

/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};

/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };

/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";

/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 5);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports) {

function logging() {
}

logging.log_level = {
  DEBUG: 1,
  INFO:  2,
  ERROR: 3
}

logging.__config__ = {
  to_console: true,
  log_level: logging.log_level.DEBUG,
  logging_element_id: null,
  max_buffer_size: 10240
}

logging.__buffer__ = ""

logging.config = function(
  logging_element_id=null, write_to_console=true, 
  log_level=logging.log_level.DEBUG, max_buffer_size=10240) {
  config = logging.__config__
  config.to_console = write_to_console
  config.log_level = log_level
  config.logging_element_id = logging_element_id
  config.max_buffer_size = max_buffer_size
}

logging._write = function(log_level, args) {
  var config = logging.__config__
  
  if (log_level >= config.log_level) {
    if (config.to_console)
      console.log.apply(console, args)

    if ((config.logging_element_id != null) && (args.length > 0)) {
      if (logging.__buffer__.length > config.max_buffer_size) {
        var len = logging.__buffer__.length
        logging.__buffer__ = logging.__buffer__.substring(len-config.max_buffer_size, len)
      }
      for (i = 0; i < args.length; i++) {
        var arg = args[i]
        if (Array.isArray(arg))
          arg = JSON.stringify(arg) 
        logging.__buffer__ += arg
      }
      logging.__buffer__ += '</br>'
      document.getElementById(config.logging_element_id).innerHTML = logging.__buffer__ 
    }
  }
}

logging.error = function() {
  logging._write(logging.log_level.ERROR, arguments)
}

logging.debug = function() {
  logging._write(logging.log_level.DEBUG, arguments)
}

logging.info = function() {
  logging._write(logging.log_level.INFO, arguments)
}

module.exports = logging;


/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

var wspipe = __webpack_require__(2)
var log = __webpack_require__(0)

client.__wspipe__ = null
client.__sid__ = 0 // channel allocated to a client over WS pipe

function client(url, consumer) {
  this.sid = client.__sid__
  client.__sid__ += 1
  this.consumer = consumer
  if (client.__wspipe__ == null) {
    client.__wspipe__ = new wspipe(url)
  }
  client.__wspipe__.subscribe(this.sid, this)
}

client.prototype._callconsumer = function(f) {
    if (typeof this.consumer !== 'undefined' && typeof this.consumer[f] === 'function') {
      var args = Array.prototype.slice.call(arguments, 1)
      this.consumer[f].apply(this.consumer, args)
    }
} 

client.prototype.onconnected = function() {
  this._callconsumer('onconnected')
}

client.prototype.ondisconnected = function() {
  this._callconsumer('ondisconnected')
}

client.prototype.init = function(exploit=false) {
  client.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'init', 'exploit': exploit})
}

client.prototype.update = function(reward, state, terminal=false) {
  client.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'update', 'reward': reward, 'state': state, 'terminal': terminal})
}

client.prototype.reset = function() {
  client.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'reset'})
}

client.prototype.onmessage = function(data) {
  switch(data['response']) {
    case 'ready':
      this._callconsumer('onready')
      break
    case 'action':
      this._callconsumer('onaction', data['data'])
      break
    case 'done':
      this._callconsumer('onresetdone')
      break
    case 'error':
      this._callconsumer('onerror', data['message'])
      break
    default:
      log.error('Unknown server response: ', data)
      break
  }
}

client.prototype.disconnect = function() {
  client.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'disconnect'})
}

module.exports = client;

/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

var log = __webpack_require__(0)

function wspipe(url) {
  this.server = url
  this.socket = null
  this.isopen = false
  this.subscribers = {}
  this.init()
}

wspipe.prototype.init = function() {
  this.socket = new WebSocket(this.server)

  this.socket.onopen = () => {
    this.isopen = true
    for (var sid in this.subscribers) {
      if (this.subscribers.hasOwnProperty(sid)) {
        this.subscribers[sid].wsopen = true
        this.subscribers[sid].onconnected()
      }
    }
  }

  this.socket.onmessage = (e) => {
    log.debug("Data received from wspipe: " + e.data)
    data = JSON.parse(e.data)
    if (this.subscribers.hasOwnProperty(data.sid)) {
      this.subscribers[data.sid].wsopen = true
      this.subscribers[data.sid].onmessage(data)
    } else {
      log.error("Data received for unidentified subscriber. Dropping that connection...")
      this.socket.send(JSON.stringify({sid:data.sid, message:'disconnect'}))
    }
  }

  this.socket.onclose = (e) => {
    this.socket = null
    this.isopen = false
    for (var sid in this.subscribers) {
      if (this.subscribers.hasOwnProperty(sid)) {
        this.subscribers[sid].wsopen = false
        this.subscribers[sid].ondisconnected()
      }
    }
    log.error("Web Socket connection closed. " + e.reason)
    log.debug("Waiting 5 sec. to try connect again...")
    setTimeout(()=>{this.init()}, 5000)
  }

  this.socket.onerror = (e) => {
    log.error("Web Socket got an error...")
  }  
}

wspipe.prototype.subscribe = function(sid, subscriber) {
  this.subscribers[sid] = subscriber
  if (this.isopen) {
    subscriber.wsopen = true
    subscriber.onconneced()
  } else {
    subscriber.wsopen = false
  }  
}

wspipe.prototype.send = function(subscriber, payload) {
  if (this.isopen && subscriber.wsopen) {
    subscriber.wsopen = false
    this.socket.send(JSON.stringify(payload))
    log.debug("Data sent through wspipe: " + JSON.stringify(payload))
  }
}

module.exports = wspipe


/***/ }),
/* 3 */
/***/ (function(module, exports, __webpack_require__) {

module.exports = {
  wspipe: __webpack_require__(2),
  client: __webpack_require__(1),
  logging: __webpack_require__(0),
  training: __webpack_require__(4)
};

/***/ }),
/* 4 */
/***/ (function(module, exports, __webpack_require__) {

var client = __webpack_require__(1)
var log = __webpack_require__(0)

function training(max_steps=10) {
  this.steps = max_steps
  this.agent_url = 'ws://127.0.0.1:9000'
  log.info('Connecting to Agent through Web Sockets proxy on ' + this.agent_url)
  this.agent = new client(this.agent_url, this)
}

training.prototype.onconnected = function() {
  this.current_step = 0
  log.info('Initializing agent...')
  this.agent.init()
}

training.prototype.onready = function() {
  log.info('Agent was initialized and ready for training')
  this.step(null)
}

training.prototype.onaction = function(action) {
  log.info('Received action: ', action)
  reward = Math.random()
  this.step(reward)
}

training.prototype.step = function (reward) {
  if (this.current_step < this.steps) {
    if (Math.random() >= 0.5)
      state = [1, 0]
    else
      state = [0, 1]
    log.info('Updating Agent with reward: ', reward, ' and state: ', state)
    this.agent.update(reward, state, false)
    this.current_step += 1
  } else {
    log.info('Training completed')
    this.stop()
  }
}

training.prototype.onerror = function(message) {
  log.error('Received error: ' + message)
  this.stop()
}

training.prototype.stop = function () {
  log.info('Disconnecting from the Agent')
  this.agent.disconnect()
}

module.exports = training;


/***/ }),
/* 5 */
/***/ (function(module, exports, __webpack_require__) {

var app = __webpack_require__(3)
app.logging.config('logging')
app.logging.info("Starting training process...")
window.training = new app.training(11)


/***/ })
/******/ ]);