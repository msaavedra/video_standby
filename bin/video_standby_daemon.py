#!/usr/bin/env python3


from video_standby.daemon import Daemon

if __name__ == '__main__':
    daemon = Daemon()
    daemon.run()