
function bandit() {
  // List out our bandits
  // Default bandit 4 (index#3) is set to most often provide a positive reward.
  this.slots = [0.2, 0.5, 0.8, 0.0]
}

bandit.prototype.pull = function(action) { 
  result = Math.random()
  if(result > this.slots[action])
    return 1
  else
    return -1
}

module.exports = bandit;
