CortexSort: Vision-Language Model Guided Robotic Manipulation with Zero-Shot Object Understanding (PyBullet + CLIP + Flask)
Developed and fine-tuned a Vision-Language Model (VLM) pipeline integrating CLIP ViT-B/32 for zero-shot text-image reasoning, enabling a simulated Franka Panda 7-DOF robot arm to interpret natural-language commands, identify target objects through Cosine Similarity Scoring, and execute autonomous pick-sort-transfer operations in real time.
Engineered a complete VLM-to-action robotics stack combining Segmentation-Mask Feature Extraction, Vision-Language Embedding Alignment, Inverse Kinematics Planning, and Constraint-Based Grasping — bridging the gap between language understanding and physical manipulation. Deployed an interactive web platform with a ChatGPT-style interface, live MJPEG simulation streaming, and multi-angle camera control for real-time human-in-the-loop VLM evaluation and robot teleoperation. 


CortexSort — Talk to a Robot, Watch it Sort 🤖

Built a system where you type "pick the red cube" and a robot arm actually does it.

How it works: CLIP (Vision-Language Model) reads your command, matches it against live camera crops of every object in the scene, and the Franka Panda arm picks the highest-scoring match — all in real time.

Stack: PyBullet · CLIP ViT-B/32 · Flask · MJPEG Streaming

What you see in the video:
→ ChatGPT-style chat window on the left
→ Live robot simulation on the right
→ Natural language in, physical manipulation out

No hardcoded rules. No object IDs. Just language and vision.

#VLM #Robotics #CLIP #ComputerVision #NLP #PyBullet #AI #MachineLearning #OpenAI #RobotManipulation

