# Configuration file for lab.
import os

import psutil

c = get_config()  # noqa

# jupyter-resource-usage extension
if "MEM_LIMIT" not in os.environ:
    # Allow 2GB buffer
    total_memory = psutil.virtual_memory().total
    mem_limit = total_memory - (2 * 1024**3)
    os.environ["MEM_LIMIT"] = str(mem_limit)

# Display memory limit
c.ServerApp.ResourceUseDisplay.mem_limit = int(os.environ["MEM_LIMIT"])

# Flash the warning to the user when they are within 10% of the memory limit
c.ServerApp.ResourceUseDisplay.mem_warning_threshold = 0.1

# Display host information from the Kernel Usage sidebar
c.ServerApp.ResourceUseDisplay.show_host_usage = True

# Enable CPU usage tracking
c.ServerApp.ResourceUseDisplay.track_cpu_percent = True

# Enable disk usage tracking
c.ServerApp.ResourceUseDisplay.track_disk_usage = True

# When to flag disk usage warning
c.ServerApp.ResourceUseDisplay.disk_warning_threshold = 0.1

# Disable token
c.ServerApp.token = ""

# The IP address the Jupyter server will listen on
c.ServerApp.ip = "0.0.0.0"

# Port the server will listen on
if "JUPYTER_PORT" not in os.environ:
    os.environ["JUPYTER_PORT"] = "8888"
c.ServerApp.port = int(os.environ["JUPYTER_PORT"])

# Log level
c.ServerApp.log_level = "DEBUG"

# Do not open browser
c.ServerApp.open_browser = False

## Allow access to hidden files
#  Default: False
c.ContentsManager.allow_hidden = True

## Allow access to hidden files
#  See also: ContentsManager.allow_hidden
c.FileContentsManager.allow_hidden = True