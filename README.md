# Intelligent-IoT-Security-Frameworks

This is the coding practise for AI on IoT security, or the state-of-the art machine learning and deep learning techniques to construct an intelligent security model for the Internet of Things.

<a href="https://github.com/goldboy225/research-material/stargazers"><img src="https://img.shields.io/github/stars/goldboy225/Intelligent-IoT-Security-Frameworks" alt="Stars Badge"/></a>
<a href="https://github.com/goldboy225/research-material/network/members"><img src="https://img.shields.io/github/forks/goldboy225/Intelligent-IoT-Security-Frameworks" alt="Forks Badge"/></a>
<a href="https://github.com/goldboy225/research-material/pulls"><img src="https://img.shields.io/github/issues-pr/goldboy225/Intelligent-IoT-Security-Frameworks" alt="Pull Requests Badge"/></a>
<a href="https://github.com/goldboy225/research-material/issues"><img src="https://img.shields.io/github/issues/goldboy225/Intelligent-IoT-Security-Frameworks" alt="Issues Badge"/></a>
<a href="https://github.com/goldboy225/research-material/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/goldboy225/Intelligent-IoT-Security-Frameworks?color=2b9348"></a>
![](https://visitor-badge.glitch.me/badge?page_id=goldboy225/Intelligent-IoT-Security-Frameworks)

Don't forget to hit the :star: if you like this repo.

This is the coding practise for AI on IoT security, or the state-of-the art machine learning and deep learning techniques to construct an intelligent security model for the Internet of Things.
There will be multiple models being or to be constructed based on many considerations, such as the infrastructure of IoT (edge, fog, or central cloud), various attacks for each layer or application (such as botnet, phishing, ransom, etc.), various publicly published datasets (such as NSL-KDD, BoT-IoT, CIC-IoT, etc.), various learning algorithms (Decision Tree, SVM, Naive Bayes, CNN, etc.), various running environments (Google Colab, Jupyter, Kaggle, etc.), and so on.
The purpose of this repo is to inspire new ideas, implement coding practises, and construct and verify models while keeping pace with the state-of-the-art solutions proposed by academics and industries. So, besides creating models, writing about the interesting ideas of other academics will also be provided.

# Contents
0. [Daily Literature Review](https://github.com/goldboy225/PhD-Journey/blob/main/Literatures%20Daily.md) This is the repository to save the recent new literatures collected, skimmed, and read, to keep up with pace of the-state-of-the art works.
1. [General Knowledge on Cybersecurity](#general-knowledge-on-cybersecurity)
2. [Datasets for Security](#datasets-for-security)  
3. [The State of the Art Studies](https://github.com/goldboy225/Intelligent-IoT-Security-Frameworks/blob/main/The-State-of-The-Art-Studies.md)  
4. [Attack Types](#attack-types) 
5. [Metaheuristic Algorithms](#metaheuristic-algorithms)

## General Knowledge on Cybersecurity
[What is Cyber Security?](https://www.kaspersky.com/resource-center/definitions/what-is-cyber-security) Cyber security is the practice of defending computers, servers, mobile devices, electronic systems, networks, and data from malicious attacks. It's also known as information technology security or electronic information security.

## Datasets for Security
| Dataset | Focus | Source | Introduction | Studies |
|--|--|--|--|--|
| [ADFA IDS](https://research.unsw.edu.au/projects/adfa-ids-datasets)|HIDS|UNSW|The datasets cover both Linux and Windows; they are designed for evaluation by system call based HIDS. |[Khater et al.,(2021)](https://www.mdpi.com/2079-9292/10/14/1633)|
| [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/index.html)|Cyber Security|UNB|Canadian Institute for Cybersecurity datasets are used around the world by universities, private industry, and independent researchers. We maintain an interactive map indicating datasets downloaded by country. ||
| [Industrial Control System (ICS) Cyber Attack Datasets](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets)|Cyber Security for IIoT|The University of Alabama in Huntsville|My primary research area is computer security. My current research involves security for Industrial Control Systems and other Cyber Physical Systems (CPS). Together with a group of student researchers, I am investigate modeling and simulation of  CPS  systems for cybersecurity research.  We primarily examine vulnerabilities, security controls, and methods for event detection.| [Prof. Tommy Morris](https://sites.google.com/a/uah.edu/tommy-morris-uah/home?authuser=0) |

## The State of the Art Studies
| Ref | Title | Idea | Year |
|--|--|--|--|
| [Fatani et al.](https://www.mdpi.com/1424-8220/22/1/140) | Advanced feature extraction and selection approach using deep learning and aquila optimizer for iot intrusion detection system | A new feature extraction and selection methods and for the IDS system using the advantages of the swarm intelligence (SI) algorithms | 2022 |
| [Disha and Waheed](https://cybersecurity.springeropen.com/articles/10.1186/s42400-021-00103-8) | Performance analysis of machine learning models for intrusion detection system using Gini Impurity-based Weighted Random Forest (GIWRF) feature selection technique | As the performance of IDS deteriorates with a high dimensional feature vector, an optimum set of features was selected through a Gini Impurity-based Weighted Random Forest (GIWRF) model as the embedded feature selection technique. | 2022 |


## Attack Types
| Attack types | Source | Introduction | Severity |
|--|--|--|--|
[DDoS Attack](https://www.javatpoint.com/what-is-ddos-attack) | javatpoint |A Distributed Denial of Service (DDoS) attack attempts to make an online service or a website unavailable by overloading it with vast floods of internet traffic generated from multiple sources. Exploited machines can include computers and other networked resources such as IoT devices. A Denial of Service (DoS) attack, in which one computer and one Internet connection are used to flood a targeted resource with packets, but a DDoS attack uses many computers and many Internet connections, often distributed globally in what is referred to as a botnet. A large-scale volumetric DDoS attack can generate traffic measured in tens of Gigabits (and even hundreds of Gigabits) per second. A regular network will not be able to handle such traffic. Attackers build a network of hacked machines known as botnets by spreading malicious code through emails, websites, and social media. Once these computers are infected, they can be controlled remotely, without their owners' knowledge, and used as an army to launch an attack against any target.|High|


## Metaheuristic Algorithms
| HM Algorithms | Study | Introduction |
|--|--|--|
| Firefly algorithm | [Xin-She Yang and Xingshi He](https://doi.org/10.1504/IJSI.2013.055801) | In this paper, we will briefly review the fundamentals of firefly algorithm together with a selection of recent publications. Then, we discuss the optimality associated with balancing exploration and exploitation, which is essential for all metaheuristic algorithms. By comparing with intermittent search strategy, we conclude that metaheuristics such as firefly algorithm are better than the optimal intermittent search strategy. We also analyse algorithms and their implications for higherdimensional optimisation problems. |
|Particle swarm optimization|Kennedy, J.; Eberhart, R.|Particle swarm optimization. In Proceedings of the ICNN’95-International Conference on Neural Networks, Perth, WA, Australia, 27 November–1 December 1995; Volume 4, pp. 1942–1948.|
|The whale optimization algorithm|[Mirjalili, S.; Lewis](https://doi.org/10.1016/j.advengsoft.2016.01.008)|This paper proposes a novel nature-inspired meta-heuristic optimization algorithm, called Whale Optimization Algorithm (WOA), which mimics the social behavior of humpback whales. The algorithm is inspired by the bubble-net hunting strategy. WOA is tested with 29 mathematical optimization problems and 6 structural design problems. Optimization results prove that the WOA algorithm is very competitive compared to the state-of-art meta-heuristic algorithms as well as conventional methods. The source codes of the WOA algorithm are publicly available at http://www.alimirjalili.com/WOA.html|
|Moth-flame optimization algorithm|[Mirjalili, S.](http://dx.doi.org/10.1016/j.knosys.2015.07.006)|In this paper a novel nature-inspired optimization paradigm is proposed called Moth-Flame Optimization (MFO) algorithm. The main inspiration of this optimizer is the navigation method of moths in nature called transverse orientation. Moths fly in night by maintaining a fixed angle with respect to the moon, a very effective mechanism for travelling in a straight line for long distances. However, these fancy insects are trapped in a useless/deadly spiral path around artificial lights. This paper mathematically models this behaviour to perform optimization. The MFO algorithm is compared with other well-known nature-inspired algorithms on 29 benchmark and 7 real engineering problems. The statistical results on the benchmark functions show that this algorithm is able to provide very promising and competitive results. Additionally, the results of the real problems demonstrate the merits of this algorithm in solving challenging problems with constrained and unknown search spaces. The paper also considers the application of the proposed algorithm in the field of marine propeller design to further investigate its effectiveness in practice. Note that the source codes of the MFO algorithm are publicly available at http://www.alimirjalili.com/MFO.html|

![image](https://user-images.githubusercontent.com/22785858/214537437-3a423506-af86-4ed2-9244-9f099cbcff8e.png)

## Contribution 🛠️
Please create an [Issue](https://github.com/goldboy225/Intelligent-IoT-Security-Frameworks/issues) for any improvements, suggestions or errors in the content.

![](https://visitor-badge.glitch.me/badge?page_id=goldboy225/Intelligent-IoT-Security-Frameworks)
