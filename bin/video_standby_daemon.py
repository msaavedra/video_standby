#!/usr/bin/env python3


from video_standby.daemon import WebDaemon

if __name__ == '__main__':
    daemon = WebDaemon()
    daemon.run()