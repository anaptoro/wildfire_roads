# ready-to-use AOIs
AOIS = {
    "Lake_Oswego_OR": (-122.709, 45.408, -122.676, 45.431),
    "Beaverton_OR_2": (-122.872, 45.493, -122.842, 45.513),
    "SandySprings_GA":(-84.377, 33.930, -84.345, 33.952),
}


DEFAULTS = dict(
    UPSCALE=2.0, # multiplies the image size before inference, higher- better recall, slower to train. lower - faster, but we may miss small crowns
    PATCH=800,# sliding window size, high better context, smaller faster but may mss the edge
    OVERLAP=0.2,  # overlap between tiles
    IOU=0.4, #lower its more agressive on merging, higher can lead to duplicates
    CONF=0.35, # confidence score
    NDVI_MIN=0.25,
    AREA_M2=(20, 600), # drop tiny shrubs & huge blobs
    ASPECT=(0.4, 2.5), # shape filter, marrow - stricter, wider, may keep odd shapes, for trees the ideal is: 0,4 -2.5
    SEG_LEN_M=50.0,
    BUFFER_M=30.0,   # for highways buffer
)

