import type { LucideIcon } from "lucide-react";
import {
  FolderGit2,
  Home,
  Image as ImageIcon,
  Users,
  Video,
  Settings,
} from "lucide-react";

export type NavItem = {
  key: string;
  titleKey: string;
  url: string;
  icon: LucideIcon;
};

export const mainNavItems: NavItem[] = [
  {
    key: "timeline",
    titleKey: "sidebar.timelineView",
    url: "#",
    icon: Home,
  },
  {
    key: "photos",
    titleKey: "sidebar.photos",
    url: "#",
    icon: ImageIcon,
  },
  {
    key: "videos",
    titleKey: "sidebar.videos",
    url: "#",
    icon: Video,
  },
  {
    key: "faceGrouping",
    titleKey: "sidebar.faceGrouping",
    url: "#",
    icon: Users,
  },
  {
    key: "manageFolders",
    titleKey: "sidebar.manageFolders",
    url: "#",
    icon: FolderGit2,
  },
];

export const settingsNavItem: NavItem = {
  key: "settings",
  titleKey: "sidebar.settings",
  url: "#",
  icon: Settings,
};

