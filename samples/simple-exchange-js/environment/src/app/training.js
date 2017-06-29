var client = require('../lib/client.js')
var log = require('../lib/logging.js')

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
