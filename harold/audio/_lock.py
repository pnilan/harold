"""Shared playback lock for audio output.

Serializes sd.play()/sd.wait() across speaker and ping modules to prevent
one stream from clobbering another when both try to play simultaneously.
"""

import threading

playback_lock = threading.Lock()
