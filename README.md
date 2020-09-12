# WGAN_GP_with_FEEDBACK
Added Gradient penalty and feedback to the Generator from Discriminator


## Modifications to the original WGAN implementation:
1. Added feedback from the discriminator in the form of attention
2.Added Gradient penalty from WGAN_GP implementation
(Note that simply adding Gradient penalty to enforce the Lipschitz constraint does not drastically improve
results in the original WGAN implementation. You can see this by removing batch normalization in WGAN implementation
with added Gradient Penalty)


## Execute
python main_grad.py --dataset lsun --dataroot /workspace/lsun --cuda --fdback --noBN --save_dir samples_gp_noBN_fd 


## Notes
Need the lsun data.
Follow the instruction in original implementation [link](https://github.com/martinarjovsky/WassersteinGAN)

## Sources
WGAN Model based on original paper [link](https://arxiv.org/abs/1701.07875).
WGAN_GP for added loss for gradient penalty  [link](https://arxiv.org/abs/1704.00028).

Cross Attention/ Feedback is my own implementation


## To-Do
Rerun and recheck the results
