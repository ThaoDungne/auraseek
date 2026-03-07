import {
  Image as ImageIcon,
  History,
  Library,
  Star,
  Users,
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
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Settings } from "lucide-react";
import { useState } from "react";
import { SettingsModal } from "@/components/common/SettingsModal";

const mainItems = [
  { title: "Ảnh", url: "#", icon: ImageIcon, key: "timeline" },
  { title: "Tin mới", url: "#", icon: History, key: "recent" },
];

const collections = [
  { title: "Album", url: "#", icon: Library, key: "albums" },
  { title: "Ảnh yêu thích", url: "#", icon: Star, key: "favorites" },
  { title: "Người", url: "#", icon: Users, key: "people" },
];


export function AppSidebar({ activeKey = "timeline", onNavClick }: { activeKey?: string, onNavClick?: (key: string) => void }) {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <Sidebar variant="inset" className="border-r-0 bg-background">
      <SidebarHeader className="h-14 flex items-center px-4 pt-4">
        <div className="flex items-center gap-2 px-2 w-full">
          <span className="font-bold text-xl tracking-tight text-foreground">AuraSeek</span>
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
                      className={`rounded-full h-11 px-4 transition-all ${isActive ? 'bg-primary/10 text-primary font-medium hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground hover:text-foreground'}`}
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

        <SidebarGroup className="px-0 py-2">
          <SidebarGroupLabel className="px-4 text-xs font-medium text-muted-foreground/70 mb-1 h-8">Bộ sưu tập</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="gap-1">
              {collections.map((item) => {
                const isActive = activeKey === item.key;
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      className={`rounded-full h-11 px-4 transition-all ${isActive ? 'bg-primary/10 text-primary font-medium hover:bg-primary/20' : 'hover:bg-muted text-muted-foreground hover:text-foreground'}`}
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

        {/* Removed Tiện ích (Tools) block as not currently functional */}

      </SidebarContent>
      <SidebarFooter className="p-4">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              onClick={() => setShowSettings(true)}
              className="rounded-full h-10 px-4 hover:bg-muted text-muted-foreground hover:text-foreground"
            >
              <Settings className="w-[1.125rem] h-[1.125rem]" />
              <span>Cài đặt</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SettingsModal open={showSettings} onOpenChange={setShowSettings} />
    </Sidebar>
  );
}
