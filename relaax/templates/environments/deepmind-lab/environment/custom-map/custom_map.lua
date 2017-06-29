local make_map = require 'custom-map/make_map'
local pickups = require 'custom-map/pickups'
local api = {}

function api:start(episode, seed)
  make_map.seedRng(seed)
  api._count = 0
end

function api:commandLine(oldCommandLine)
  return make_map.commandLine(oldCommandLine)
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function readFile(file)
    local f = io.open(file, "rb")
    local content = f:read("*all")
    f:close()
    return content
end

function api:nextMap()
  api._count = api._count + 1
  map = readFile('/app/environment/custom-map/t_maze')
  return make_map.makeMap("t_maze_" .. api._count, map)
end

return api
