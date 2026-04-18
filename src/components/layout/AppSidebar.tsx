import {
  Image as ImageIcon,
  Film,
  Library,
  Star,
  Users,
  Lock,
  Trash2,
  CopySlash,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Settings, ChevronRight } from "lucide-react";
import { useState } from "react";
import { SettingsModal } from "@/components/common/SettingsModal";

const mainItems = [
  { title: "Ảnh", url: "#", icon: ImageIcon, key: "timeline" },
  { title: "Video", url: "#", icon: Film, key: "videos" },
];

const collections = [
  { title: "Album", url: "#", icon: Library, key: "albums" },
  {
    title: "Yêu thích", url: "#", icon: Star, key: "favorites",
    subItems: [
      { title: "Ảnh yêu thích", key: "favorite_photos" },
      { title: "Video yêu thích", key: "favorite_videos" },
    ]
  },
  { title: "Người", url: "#", icon: Users, key: "people" },
  {
    title: "Khử trùng lặp", url: "#", icon: CopySlash, key: "duplicates",
    subItems: [
      { title: "Ảnh trùng lặp", key: "duplicate_images" },
      { title: "Video trùng lặp", key: "duplicate_videos" },
    ]
  },
];

const management = [
  { title: "Thư mục ẩn", url: "#", icon: Lock, key: "hidden" },
  { title: "Thùng rác", url: "#", icon: Trash2, key: "trash" },
];


export function AppSidebar({
  activeKey = "timeline",
  onNavClick,
  sourceDir = "",
  onSourceDirChange,
}: {
  activeKey?: string;
  onNavClick?: (key: string) => void;
  sourceDir?: string;
  onSourceDirChange?: (dir: string) => void;
}) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <Sidebar variant="inset" className="border-r-0 bg-background">
      <SidebarHeader className="h-16 flex items-center px-6 pt-5">
        <div className="flex items-center gap-2 w-full">
          <span className="font-extrabold text-2xl tracking-tighter text-foreground">AuraSeek</span>
        </div>
      </SidebarHeader>
      <SidebarContent className="px-3 gap-0 mt-2">

        <SidebarGroup className="px-0 py-2">
          <SidebarGroupContent>
            <SidebarMenu className="gap-1">
              {mainItems.map((item) => {
                const isActive = activeKey === item.key;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={`rounded-full h-11 px-4 transition-all text-[14px] ${isActive ? 'bg-primary/10 text-primary font-bold hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground/80 hover:text-foreground font-medium'}`}
                      onClick={() => onNavClick?.(item.key)}
                    >
                      <a href={item.url} className="flex items-center gap-4">
                        <item.icon className="w-[1.125rem] h-[1.125rem]" />
                        <span>{item.title}</span>
                      </a>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="px-0 py-3 mt-2">
          <SidebarGroupLabel className="px-6 text-[11px] font-extrabold text-muted-foreground/40 uppercase tracking-[0.15em] mb-2 h-6">Bộ sưu tập</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="gap-1">
              {collections.map((item) => {
                const isGroupActive = activeKey === item.key || item.subItems?.some(s => activeKey === s.key);

                if (item.subItems) {
                  return (
                    <Collapsible
                      key={item.title}
                      asChild
                      defaultOpen={isGroupActive}
                      className="group/collapsible"
                    >
                      <SidebarMenuItem>
                        <CollapsibleTrigger asChild>
                          <SidebarMenuButton
                            className={`rounded-full h-11 px-4 transition-all text-[14px] ${isGroupActive ? 'bg-primary/10 text-primary font-bold hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground/80 hover:text-foreground font-medium'}`}
                            tooltip={item.title}
                          >
                            <item.icon className="w-[1.125rem] h-[1.125rem]" />
                            <span>{item.title}</span>
                            <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                          </SidebarMenuButton>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <SidebarMenuSub className="ml-5 mt-1 border-l-2 border-border/40 pl-3">
                            {item.subItems.map((sub) => (
                              <SidebarMenuSubItem key={sub.key}>
                                <SidebarMenuSubButton
                                  isActive={activeKey === sub.key}
                                  onClick={() => onNavClick?.(sub.key)}
                                  className={`rounded-full px-3 py-1.5 cursor-pointer text-[13px] ${activeKey === sub.key ? 'text-primary font-bold' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'}`}
                                >
                                  {sub.title}
                                </SidebarMenuSubButton>
                              </SidebarMenuSubItem>
                            ))}
                          </SidebarMenuSub>
                        </CollapsibleContent>
                      </SidebarMenuItem>
                    </Collapsible>
                  );
                }

                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={`rounded-full h-11 px-4 transition-all text-[14px] ${isGroupActive ? 'bg-primary/10 text-primary font-bold hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground/80 hover:text-foreground font-medium'}`}
                      onClick={() => onNavClick?.(item.key)}
                    >
                      <a href={item.url} className="flex items-center gap-4">
                        <item.icon className="w-[1.125rem] h-[1.125rem]" />
                        <span>{item.title}</span>
                      </a>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="px-0 py-3">
          <SidebarGroupLabel className="px-6 text-[11px] font-extrabold text-muted-foreground/40 uppercase tracking-[0.15em] mb-2 h-6">Quản lý</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="gap-1">
              {management.map((item) => {
                const isActive = activeKey === item.key;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={`rounded-full h-11 px-4 transition-all text-[14px] ${isActive ? 'bg-primary/10 text-primary font-bold hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground/80 hover:text-foreground font-medium'}`}
                      onClick={() => onNavClick?.(item.key)}
                    >
                      <a href={item.url} className="flex items-center gap-4">
                        <item.icon className="w-[1.125rem] h-[1.125rem]" />
                        <span>{item.title}</span>
                      </a>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setShowSettings(true)}
              className="rounded-full h-11 px-4 hover:bg-muted text-muted-foreground/80 hover:text-foreground text-[14px] font-medium"
            >
              <Settings className="w-[1.125rem] h-[1.125rem]" />
              <span>Cài đặt</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SettingsModal
        open={showSettings}
        onOpenChange={setShowSettings}
        currentSourceDir={sourceDir}
        onSourceDirChange={onSourceDirChange}
      />
    </Sidebar>
  );
}
