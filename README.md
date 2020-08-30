# What does this repository contain ?
Scripts that help in website fingerprinting.

# What is TOR ?
The Onion Router TOR is a low latency anonymity network of relays which aims to hide the
recipient and the content of communications from a local observer, i.e., an entity that can
eavesdrop the traffic between the client and the server. Based on that research, it offers a
technology that bounces internet users' and websites' traffic through "relays" run by thousands
of volunteers around the world, making it extremely hard for anyone to identify the source of
the information or the location of the user. It is extremely useful in totalitarian regimes as it
helps users to freely communicate on the web.

# What is Website Fingerprinting ?
Website Fingerprinting is an attack on the anonymity of the TOR Network. Initially developed
for security purposes, website fingerprinting (also known as device fingerprinting) is a tracking
technique capable of identifying individual users based on their browser and device settings. In
order for websites to display correctly, your browser makes certain information available about
your device, including your screen resolution, operating system, location, and language
settings. These details essentially make up the ridges of your digital fingerprint. It helps the
attacker to know about the web history of a client and which page the client is browsing through
encrypted or anonymization networks.

# Workflow

1. It can be concluded that the TOR anonymity network is prone to website fingerprinting, analyse the effect of website
fingerprinting. 

2. This research focusses on a website fingerprinting technique, K-Fingerprinting,
proposed by Hayes that is based on Random Forests and tries to improve the results of the
existing state-of-the-art system. 

3. The dataset assumes a TOR Hidden Service of 30 websites
only. Each Website further contains 100 webpages which are split in 80:20 for the training and the testing set.

4. In the first step previous system is transferred to a  model that uses the Random
Forest Classifier. In the subsequent steps, work is done on on selecting and computing the
features in order to arrive at a conclusion.
