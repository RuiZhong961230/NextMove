# NextMove
Next Point of Interest (POI) Recommendation System Driven by User Probabilistic Preferences and Temporal Regularities


## Abstract
The Point of Interest (POI) recommendation system is a critical tool for enhancing user experience by analyzing historical behaviors, social network data, and real-time location information with the increasing demand for personalized and intelligent services. 

However, existing POI recommendation systems face three major challenges: 

- (1) oversimplification of user preference modeling, limiting adaptability to dynamic user needs; 

- (2) lack of explicit arrival time modeling, leading to reduced accuracy in time-sensitive scenarios; 

- (3) complexity in trajectory representation and spatiotemporal mining, posing difficulties in handling large-scale geographic data. 

This paper proposes NextMove, a novel POI recommendation model that integrates four key modules to address these issues. 
Specifically, the Probabilistic User Preference Generation Module (PUPG) first employs Latent Dirichlet Allocation and a user preference network to model user personalized interests dynamically by capturing latent geographical topics. 
Secondly, the Self-Attention-based Arrival Time Prediction Module (SATP) utilizes a Multi-Head Attention Mechanism to extract time-varying features, improving the precision of arrival time estimation. 
Thirdly, the Transformer-based Trajectory Representation Module (TTRM) encodes sequential dependencies in user behavior, effectively capturing contextual relationships and long-range dependencies for accurate future location forecasting. 
Finally, the Next Location Feature-Aggregation Module (NLFA) integrates the extracted representation features through an FC-based nonlinear fusion mechanism to generate the final POI recommendation. Extensive experiments conducted on real-world datasets demonstrate the superiority of the proposed NextMove over state-of-the-art methods. These results validate the effectiveness of NextMove in modeling dynamic user preferences, enhancing arrival time prediction, and improving POI recommendation accuracy.

## Acknowledgments
Our model is built based on the model of [MCLP](https://github.com/SUNSTARK/MCLP) and [ImNext](https://github.com/simplehx/ImNext).
