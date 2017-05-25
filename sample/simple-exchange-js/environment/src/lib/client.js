var wspipe = require('./wspipe.js')
var log = require('./logging.js')

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