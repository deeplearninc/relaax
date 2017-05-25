var log = require('./logging.js')

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
