---
layout: default
title: Publications
---

{% for pub in site.data.publications %}
[{{pub.title}}]({{pub.link}}): {{pub.tagline}} {{pub.authors}}. {{pub.venue}}
{% endfor %}

