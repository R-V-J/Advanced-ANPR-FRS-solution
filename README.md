# Automatic Number Plate Recognition (ANPR) and Facial Recognition
#### Design and develop a technological solution that can accurately perform the Automatic Number Plate Recognition (ANPR) along with Facial Recognition from the available CCTV feeds. The solution should be able to recognize number plates that are written in typical non-standard ways using varying font styles, sizes, designs, symbols, languages etc., i.e. difficult to recognize by existing ANPR Systems.

<summary>Table of Contents</summary>

- [Description](#description)
- [Tech Stack](#tech-stack)
- [Scope](#scope)
- [Methods](#methods)
- [Face Detection In OpenCV](#face-detection-in-opencv)
- [EasyOCR](#easyocr)
- [Team Members](#team-members)
- [Links](#links)

## 📝Description
This is one of the unique projects done by a team of six members which inludes the domain of **video analytics and CCTV**. In this prroject, we extensively used the **computer vision and the image processing** technologies using various in-built modules and libraries of **Python** programming language. As a part of our attempt to the **KAVACH 2023 - The Cybersecurity Hackathon**, we tried to advance the existing technologies by adding few valuable edits.

## 🤖Tech-Stack
<img src="https://user-images.githubusercontent.com/119912993/232730672-d59fe837-5073-48f5-8fc0-6d98ff0bccfe.png" height=100 width=100> &nbsp;
<img src="https://user-images.githubusercontent.com/119912993/231549351-5a7fcfdc-998a-4880-a2af-7225c80ccbc7.png" height=100 width=100> &nbsp;
<img src="https://user-images.githubusercontent.com/119912993/231549738-5013b6f7-3754-4330-95d9-adc68a294fe0.png" height=100 width=200> &nbsp;
<img src="https://user-images.githubusercontent.com/119912993/231636358-72a49f70-5d57-4875-a0f6-e6251f0453eb.png" height=100 width=100> &nbsp;
<img src="https://user-images.githubusercontent.com/119912993/231635764-17335a4a-c329-43a0-b90e-e4e81499e9e8.png" height=100 width=100> &nbsp;
<img src="https://user-images.githubusercontent.com/119912993/231635902-0243d3d5-993a-458e-b232-19cb46b0d37d.png" height=100 width=100> &nbsp;

## 🔮Scope
- **Automatic Number Plate Recognition (ANPR):** The system will accurately recognize and extract number plate information from CCTV feeds, including non-standard number plates with varying font styles, sizes, designs, symbols, languages, and other atypical characteristics.
- **Facial Recognition:** The system will incorporate advanced Facial Recognition technology to accurately identify and match faces captured from CCTV feeds against known databases, enabling effective identification and tracking of individuals.
- **Integration with CCTV Feeds:** The system will seamlessly integrate with existing CCTV infrastructure, leveraging real-time ANPR and Facial Recognition processing from available feeds.
- **Machine Learning Algorithms:** Precise machine learning algorithms will be utilized to train the system on diverse data sets, enabling it to adapt and accurately recognize number plates and faces even in challenging scenarios.
- **User-friendly Interface:** The system will feature an intuitive and user-friendly interface for easy configuration, monitoring, and management of ANPR and Facial Recognition functionalities, empowering operators to efficiently operate and manage the system.
- **Scalability and Flexibility:** The solution will be designed to be scalable and flexible, allowing for easy integration with existing security systems and customization to meet the specific requirements of different environments adhering to their use cases.
- **Security and Privacy:** The system will prioritize robust security and privacy measures to protect the integrity and confidentiality of captured data, comply with relevant data protection regulations, and ensure secure access and usage of the system.
- **Testing and Validation:** The solution will undergo rigorous testing and validation to ensure its accuracy, reliability, and performance in accurately identifying vehicles and individuals, including non-standard number plates, in various real-world conditions.
- **Support and Maintenance:** The system’s comprehensive support and maintenance services ensure smooth operation, timely updates, and ongoing technical assistance to address any issues or updates that may arise during the system's lifecycle.

## ⏩Methods
- ***Data Acquisition:*** The system captures video or image data from CCTV feeds, which may include images of vehicles with number plates and faces of individuals.
- ***ANPR Processing:*** The system uses ANPR algorithms to accurately recognize and extract number plate information from the captured data, even if the number plates have non-standard fonts, designs, or other atypical characteristics. This involves image preprocessing, feature extraction, and pattern recognition techniques to accurately identify the number plates.
- ***Facial Recognition Processing:*** The system employs Facial Recognition algorithms to analyze the facial features of individuals captured from the CCTV feeds. This involves detecting and extracting facial features such as eyes, nose, mouth, and face shape, and comparing them against known databases to identify and match faces.
- ***Data Integration:*** The system integrates the recognized number plate information and facial recognition results, associating them with the corresponding vehicle and individual identities, if available in the databases.
- ***Decision Making:*** The system uses the integrated data to make decisions, such as determining whether a recognized number plate matches with a known vehicle or if a recognized face matches with a known individual in the database.
- ***Alert Generation:*** If a match is found or if any other predefined criteria are met, the system generates alerts or notifications to operators or other designated personnel for further action.

## 🕵️‍♂️Face Detection In OpenCV
- **OpenCV Implementation**:  it is a machine learning-based object detection algorithm. The classifier uses a small fixed dataset with 3-4 defined classes for facial recognition from web-camera input providing valuable insights and preliminary results.The smaller dataset will be a starting point for initial experimentation and testing.
- **Teachable Machine:** In previous experimentation, we explored the use of Teachable Machine. This involved investigating the feasibility of leveraging a pre-existing model trained on a larger dataset to potentially enhance the accuracy and performance of the facial recognition feature.

## 🔠🔣EasyOCR
- **Easy OCR :** Library that provides **Optical Character Recognition** (OCR) capabilities specifically for A**utomatic Number Plate Recognition** (ANPR) tasks using deep learning techniques.
- -  Using Easy OCR we utilize **Deep Learning** techniques to learn and **recognize license number plate characters** from **images** or **video streams** in order to achieve high accuracy and robustness.


## 👨‍💻Team Members
- [Arsh Khan](https://github.com/Arsh-Khan)
- [Komal Sambhus](https://github.com/Komal0103)
- [Himanshu Singh](https://github.com/Himanshu-singh04)
- [Rushi Jani](https://github.com/R-V-J)
- [Siddharth Shenoy](https://github.com/Shenoy37)
- [Aryan Karawale](https://github.com/Aryan-karawale)

## 🔗Links

- [GitHub Repository](https://github.com/R-V-J/Advanced-ANPR-FRS-solution)
- [Demo Video](https://drive.google.com/drive/u/1/folders/1yRx7JDimmQxAL-2GM5gQsnqcsfeyDL5c)
