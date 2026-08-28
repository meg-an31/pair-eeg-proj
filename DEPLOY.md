# Deployment

Live on pair-18 at **https://eeg.2-29-1-92.sslip.io**

| | |
|---|---|
| Capture page | https://eeg.2-29-1-92.sslip.io/ |
| Viewer page | https://eeg.2-29-1-92.sslip.io/view.html |
| Socket | `wss://eeg.2-29-1-92.sslip.io/ws` |

HTTPS is not optional here: Web Bluetooth requires a secure context, and a page
served over https cannot open a `ws://` socket. Caddy terminates TLS with a
Let's Encrypt certificate for the sslip.io name, which resolves to the box's
own IP, so no DNS setup is needed.

## Layout on the server

    /home/pair18/pair-eeg/       code + venv
    /home/pair18/pair-eeg/sessions/   recordings
    /srv/pair-eeg/               web files, readable by the caddy user
    /etc/systemd/system/pair-eeg.service
    /etc/caddy/Caddyfile         one appended vhost block

The web files are copied to `/srv` rather than served from the home directory
because Caddy runs as its own user and cannot traverse `/home/pair18` (0750).

## Operating it

    sudo systemctl status pair-eeg
    sudo systemctl restart pair-eeg
    journalctl -u pair-eeg -f

## Redeploying

    tar czf deploy.tgz --exclude=.venv --exclude=.git --exclude=sessions .
    scp deploy.tgz pair18@2.29.1.92:/tmp/
    ssh pair18@2.29.1.92 '
      tar xzf /tmp/deploy.tgz -C ~/pair-eeg &&
      sudo rsync -a --delete ~/pair-eeg/web/ /srv/pair-eeg/ &&
      sudo systemctl restart pair-eeg'

## Access model

One capture client, unlimited viewers. The headband allows a single BLE
connection, so a second streamer could only ever be a mistake — a stale tab, a
duplicate window. The server admits one and answers the rest with
`capture_busy` and an explanation the page shows. Viewers are read-only:
they cannot send sensor data and cannot control the session.

## Not done yet

The server is HTTP-authenticated by nobody. Anyone who finds the URL can claim
the capture slot or watch the stream. That is fine for a demo on a box whose
address is not published, and needs a token before it carries real recordings.
