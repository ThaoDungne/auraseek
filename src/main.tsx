import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import { DictionaryProvider } from "./i18n/DictionaryProvider";
import { ThemeProvider } from "./components/theme-provider";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
    <DictionaryProvider>
      <App />
    </DictionaryProvider>
  </ThemeProvider>,
);
