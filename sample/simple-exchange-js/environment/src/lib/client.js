var wspipe = require('./wspipe.js')
var log = require('./logging.js')

window.__wspipe__ = null;

function client(sid, url, consumer) {
  this.sid = sid
  this.consumer = consumer
  if (window.__wspipe__ == null) {
    window.__wspipe__ = new wspipe(url)
  }
  window.__wspipe__.subscribe(sid, this)
}

client.prototype.states = {
  none: 0,
  init: 1,
  update: 2,
  reset: 3
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
  this.state = this.states.init
  window.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'init', 'exploit': exploit})
}

client.prototype.update = function(reward, state, terminal=false) {
  this.state = this.states.update
  window.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'update', 'reward': reward, 'state': state, 'terminal': terminal})
}

client.prototype.onmessage = function(data) {
  switch(data['response']) {
    case 'ready':
      this._callconsumer('onready')
      break
    case 'action':
      this._callconsumer('onaction', data['action'])
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
  window.__wspipe__.send(this,
    {'sid': this.sid, 'command': 'disconnect'})
}

module.exports = client;