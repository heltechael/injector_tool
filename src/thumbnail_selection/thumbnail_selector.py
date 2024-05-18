class ThumbnailSelector:
    def __init__(self, config, thumbnails_by_class, load_thumbnail):
        self.config = config
        self.thumbnails_by_class = thumbnails_by_class
        self.load_thumbnail = load_thumbnail
        self.thumbnail_assessment_threshold = config.get('THUMBNAIL_ASSESSMENT_THRESHOLD')
        self.USE_BEST_THUMBNAILS = config.get('USE_BEST_THUMBNAILS')

    def select_thumbnails(self, classes):
        selected_thumbnails = {}
        for eppo_class in classes:
            if eppo_class in self.thumbnails_by_class:
                class_thumbnails = self.thumbnails_by_class[eppo_class]
                
                selected_thumbnails[eppo_class] = class_thumbnails

        return selected_thumbnails

