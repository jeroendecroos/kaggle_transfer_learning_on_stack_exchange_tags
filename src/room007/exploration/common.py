#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re


def tag_in_text(tag, text):
    tag_re = '\\b{tag}\\b'.format(tag=re.escape(tag))
    return re.search(tag_re, text) is not None


def norm_tag(tag):
    return tag.replace('-', ' ')


def question_mentions_tag(qrec, tag):
    normed = norm_tag(tag)
    return (tag_in_text(normed, qrec['title']) or
            tag_in_text(normed, qrec['content']))
