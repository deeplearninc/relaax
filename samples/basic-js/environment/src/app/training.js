var client = require('../lib/client.js')
var log = require('../lib/logging.js')
var bandit = require('./bandit.js')

function training(max_steps=3000) {
  this.steps = max_steps
  this.bandit = new bandit()
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
  log.info('Step:', this.current_step, ' action: ', action)
  this.step(this.bandit.pull(action))
}

training.prototype.step = function (reward) {
  if (this.current_step < this.steps) {
    log.debug('Updating Agent with reward: ', reward)
    this.agent.update(reward, [], false)
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
