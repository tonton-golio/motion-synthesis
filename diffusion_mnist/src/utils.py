import numpy as np
from tqdm import trange

n_samples = 1
X_rand = torch.randn((n_samples, 8))
y_rand = torch.randint(0, 10, (n_samples, 1))
print("X_rand", X_rand.shape, X_rand)
print("y_rand", y_rand.shape, y_rand)

# get average of where y is Y_rand[0]
mean_z = z[y == y_rand[0].item()].mean(axis=0)
mean_z_projected = projector.transform(mean_z.reshape(1, -1))
print("mean_z", mean_z.shape, mean_z)
print("mean_z_projected", mean_z_projected.shape, mean_z_projected)


# show on projection
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(2, 3, figure=fig)

# add axes
ax_recon = [fig.add_subplot(gs[i, 0]) for i in range(2)]
ax_prodj = fig.add_subplot(gs[:, 1:])
projection_sample = projector.transform(X_rand.cpu().detach().numpy())

scatter_prodj = ax_prodj.scatter(
    projection[:, 0], projection[:, 1], c=y, cmap="tab10", alpha=0.3, s=2
)
# color bar
fig.colorbar(scatter_prodj, ax=ax_prodj, orientation="vertical")
ax_prodj.scatter(
    projection_sample[:, 0],
    projection_sample[:, 1],
    c="black",
    alpha=1.0,
    s=200,
    marker="x",
    label=y_rand,
)


prodj_trajec_sample1 = projection_sample.copy()
trajec = X_rand[0].cpu().detach().numpy().copy()
# reconstruction
with torch.no_grad():
    decoded = utility.autoencoder.decode(X_rand)
    ax_recon[0].imshow(decoded.squeeze(), cmap="gray")
    ax_recon[0].set_title("Intitial Reconstruction")
    ax_recon[0].axis("off")

# diffusion
diffusion_steps = 10


with torch.no_grad():
    for i in trange(diffusion_steps):
        noise = torch.randn_like(X_rand) * 0.1
        pred_noise, noise = model((X_rand, y_rand))
        # print('pred_noise', pred_noise)
        X_rand = X_rand - pred_noise * 0.3

        trajec = np.vstack([trajec, X_rand.cpu().detach().numpy()])
        # add to trajectory
        # projection_sample = projector.transform(X_rand.cpu().detach().numpy())
        # print(projection_sample)
        # prodj_trajec_sample1 = np.vstack([prodj_trajec_sample1, projection_sample])

    prodj_trajec_sample1 = projector.transform(trajec)

    decoded = utility.autoencoder.decode(X_rand)
    ax_recon[1].imshow(decoded.squeeze(), cmap="gray")

    projection_sample = projector.transform(X_rand.cpu().detach().numpy())
    ax_prodj.scatter(
        projection_sample[-1, 0],
        projection_sample[-1, 1],
        c="blue",
        alpha=1.0,
        s=200,
        marker="x",
        label="diffused",
    )


# plot trajectory
ax_prodj.plot(
    prodj_trajec_sample1[:, 0],
    prodj_trajec_sample1[:, 1],
    c="black",
    alpha=0.5,
    label="Trajectory",
)
ax_prodj.legend()
