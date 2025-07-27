# Running as a lichess bot

To run the engine as a lichess bot you'll need to install the [Lichess Bot API](https://github.com/ShailChoksi/lichess-bot) and copy some files over.

If you pull [https://github.com/ShailChoksi/lichess-bot](https://github.com/ShailChoksi/lichess-bot) to the same parent directory of moonfish, such that your local setup looks like:

```shell
$ tree moonfish/ lichess-bot/ -L 1
moonfish/
├── LICENSE
...
lichess-bot/
├── README.md
...

16 directories, 17 files
```

you could then run the below to setup (build + move files) lichess to work with the engine:

don't forget to create a [`.env`](../.env.example) with your lichess API token (with bot:play scope)

```shell
lichess/setup_macos.sh
```
