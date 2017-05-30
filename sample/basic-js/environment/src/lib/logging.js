
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