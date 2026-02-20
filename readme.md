# 🖥️ Realium Liveness Microservice

**Realium** is a **biometric-based authentication system** designed for **Web3 platforms**, enabling **proof of self** and ensuring that users are genuinely present during authentication.  
The **Realium Liveness Microservice** is a **Python-based module** that verifies whether a user is physically in front of the camera, forming a critical part of Realium’s anti-spoofing and identity verification pipeline.  

It analyzes **face movements**, **texture**, and **embeddings** in **real-time** to produce a reliable verdict of:

- ✅ **Live (Verified User)**  
- ⚠️ **Spoof (Potential Fraud)**  

---

## ⚙️ How It Works

The microservice continuously monitors a camera feed using a combination of techniques:

- **CNN-based predictions**  
- **Texture analysis**  
- **Challenge-response sequences**
  

## 1️⃣ Face Detection 👤

The first step is detecting faces in the camera feed:

- Uses **facial landmark detection** to precisely locate facial features.  
- Creates a **bounding box** around the detected face and crops it for further analysis.  
- Tracks visibility using a **last_seen_time** mechanism.  
- **Resets session** if the face disappears for too long, ensuring temporary absence or camera glitches don’t compromise verification.  

---

## 2️⃣ Challenge-Based Verification 🎯

To confirm that the detected face belongs to a **live human**, the microservice generates a sequence of **challenges**:

| Challenge | Description | Icon |
|-----------|-------------|------|
| Blink | Detect eye blinking | 👁 |
| Turn head | Left or right rotation | ↩️ ↪️ |
| Smile | Smile verification | 😁 |
| Speak | Say a word like “Hello” | 🗣️ |
| DONE | Final step indicating completion | ✅ |

**Evaluation:**

- Tracks **CNN predictions**, **challenge progress**, and **texture scores**.  
- Only when all challenges are successfully completed is the face verified as **Live**, proving **proof of self**.  

---

## 3️⃣ Immediate DONE Logic ✅

When the **DONE** state is reached:

- **If thresholds met:**  
  - Marks the face as **Live Verified** ✅  
  - Captures session frames and generates **feature embeddings** for audit or future analysis.  

- **If thresholds not met:**  
  - Marks the attempt as **Spoof Detected** ⚠️  
  - Saves an image of the suspicious face for further review.  

This ensures **quick decision-making** while maintaining **accuracy**.  

---

## 4️⃣ CNN Prediction & Texture Analysis 🧠

Two key techniques are used to detect spoofing:

### CNN Prediction
- Cropped face is **normalized** and passed through a **pre-trained CNN model**.  
- CNN outputs a **live probability score** for each frame.  

### Texture Analysis (LBP)
- Uses **Local Binary Patterns (LBP)** to analyze face texture.  
- Detects **printed photos**, **screens**, or **masks**.  

✅ **Final liveness verdict** combines CNN scores and texture analysis to confirm **proof of self**.  

---

## 5️⃣ GUI Feedback 🖼️

If GUI mode is enabled:

- Shows **bounding boxes** around detected faces  
- Displays **challenge messages** for user actions  
- Shows a **progress bar** for challenge sequence completion  

After the final verdict:

- ✅ **Green banner** → Live Verified (User authenticated)  
- ⚠️ **Red banner** → Spoof Detected (Potential fraud)  

This provides **real-time understanding** of the verification process.  

---

## 🔧 Summary

The **Realium Liveness Microservice** is an essential component of **Realium**, a **biometric-based authentication system for Web3 platforms**, ensuring **proof of self** for users by combining:

- **Face detection** 👤  
- **Challenge-based verification** 🎯  
- **CNN & texture analysis** 🧠  
- **Real-time GUI feedback** 🖼️  

It is **robust**, **accurate**, and integrates seamlessly in both **live monitoring** and **automated pipelines**.

---

