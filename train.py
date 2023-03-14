import argparse, torch, clip
from stylegan2 import Generator
from coralstyle import CoralAttnNet
from criteria.loss import (
    clip_loss, id_loss,
    tv_loss, l2_loss,
    can_area_loss,
)
from torchvision import utils

parser = argparse.ArgumentParser(description="Generate samples from the generator")
parser.add_argument("--device", type=str, default='cuda', help="output image size of the generator")
parser.add_argument('--text', type=str, default='Happy', help='text to clip')
parser.add_argument("--batch", type=int, default=2, help="batch size")
parser.add_argument("--print_interval", type=int, default=2, help="print frequency")

args = parser.parse_args()
device = torch.device(args.device)
with torch.no_grad():
    text = clip.tokenize([args.text]).to(device)
g_ema = Generator(256, 512, 8, channel_multiplier=2).to(device)
ckpt = torch.load('./pretrained_models/ffhq-256x256-550000.pt', map_location=device)
g_ema.load_state_dict(ckpt["g_ema"], strict=False)
model = CoralAttnNet(g_ema).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
epoches = 20000
for epoch in range(epoches):
    z = torch.randn((args.batch, 512)).to(device)
    optim.zero_grad()
    image, image_star, image_bar, masks, delta_w_plus = model(z)
    clip_l = clip_loss(image_star, image_bar, text)
    id_l = id_loss(image, image_star)
    tv_l = tv_loss(masks)
    l2_l = l2_loss(delta_w_plus)
    can_l = can_area_loss(masks)
    loss = clip_l + id_l + tv_l + l2_l + can_l
    if epoch % args.print_interval == 0:
        print(epoch, loss.item(), clip_l.item(), id_l.item(), tv_l.item(), l2_l.item(), can_l.item())
        images = torch.cat([
            image, image_star, image_bar, (2*masks[-1]-1).repeat(1, 3, 1, 1)
        ], dim=0)
        print(images.shape)
        utils.save_image(
            images, f"sample/{str(epoch).zfill(6)}.png",
            nrow=args.batch, normalize=True, value_range=(-1, 1),
        )
    loss.backward()
    optim.step()
