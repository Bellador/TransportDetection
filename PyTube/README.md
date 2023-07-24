# YoutubeDownloader

This script is a small and simple wrapper around the great and lightweight library [PyTube](https://github.com/pytube/pytube).
It allows to easily pass a list of links that are downloaded in sequence. 

The script takes care of:

- creating a project directory where all videos are stored
- target resolution and MIME setting, by finding the stream that best fits your need
- progress displaying
- output name formatting to only include ASCII characters and replaces white-space with underscores

